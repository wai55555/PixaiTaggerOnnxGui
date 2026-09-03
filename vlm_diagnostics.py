"""接続診断（260901_VLM_spec.md 12章 / design.md 4.6節）。

「その接続設定が実際に動くか」を一括で確認する。選択画像の品質を見る機能ではない。
結果は接続定義に書き戻さず、状態キャッシュとして扱う。設定変更で無効化する。
"""
from __future__ import annotations

import io
import re
import socket
import ssl
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

from PIL import Image

from vlm_connections import VlmConnection
from vlm_errors import VlmErrorReason
from vlm_image import ImagePreprocessConfig, prepare_image
from vlm_profiles import GenerationProfile, build_system_prompt, build_user_prompt
from vlm_protocols import (
    VlmCallSpec, VlmHttpRequest, apply_connection_auth, apply_request_body,
    default_auth_key, extract_by_path,
    get_protocol, apply_request_headers,
)
from vlm_transport import RawHttpResponse, execute_http

# Cloudflare のトークン検証はアカウント ID やモデルに依存しない専用エンドポイント。
_CLOUDFLARE_TOKEN_VERIFY_URL = "https://api.cloudflare.com/client/v4/user/tokens/verify"


class DiagStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class DiagItem:
    name: str
    status: DiagStatus
    detail: str = ""


@dataclass
class DiagReport:
    connection_id: str
    items: list[DiagItem] = field(default_factory=list)
    http_status: int | None = None   # live リクエストが返した HTTP ステータス（あれば）

    def add(self, name: str, status: DiagStatus, detail: str = "") -> None:
        self.items.append(DiagItem(name, status, detail))

    def item(self, name: str) -> DiagItem | None:
        for i in self.items:
            if i.name == name:
                return i
        return None

    @property
    def overall(self) -> DiagStatus:
        if any(i.status is DiagStatus.FAIL for i in self.items):
            return DiagStatus.FAIL
        if any(i.status is DiagStatus.WARN for i in self.items):
            return DiagStatus.WARN
        return DiagStatus.PASS

    @property
    def can_mark_binding_verified(self) -> bool:
        """接続設定を「確認済み」として記録できるか。

        通常は HTTP 200 で本文抽出まで成功した場合だけ実証済みとする。ただし
        429 は認証済みのリクエストがエンドポイントへ到達したことを示すため、
        認証・リクエスト組み立て・画像入力が PASS なら到達確認済みとして扱う。
        この場合も ``overall`` は WARN のままなので、本文取得成功と混同しない。
        課金不足の 429 は HTTP を FAIL にしているため、この例外には入らない。
        """
        http = self.item("HTTP response")
        if http is None:
            return False
        extraction = self.item("Caption extraction")
        if http.status is DiagStatus.PASS and extraction is not None:
            if extraction.status is DiagStatus.PASS:
                return True
            # 200応答が診断用トークン上限で切れた場合も、モデルまで到達したことは
            # 確認できている。本文抽出成功とは区別するため、詳細に明示されたケースだけ
            # を到達確認として扱う（content policy 等の一般WARNは含めない）。
            if (extraction.status is DiagStatus.WARN
                    and "endpoint reachable" in (extraction.detail or "").lower()):
                return True
        if self.http_status != 429 or http.status is not DiagStatus.WARN:
            return False
        required = ("Auth", "Request build", "Image input")
        return all((item := self.item(name)) is not None
                   and item.status is DiagStatus.PASS for name in required)


def _tiny_test_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _first_cf_message(body: dict) -> str:
    for coll in (body.get("errors"), body.get("messages")):
        if isinstance(coll, list) and coll and isinstance(coll[0], dict):
            msg = coll[0].get("message")
            if msg:
                return str(msg)
    return ""


def _response_error_detail(raw: RawHttpResponse) -> str:
    """プロバイダーのJSONエラーを、キー値を含めず診断画面へ返す。"""
    body = raw.json_body if isinstance(raw.json_body, dict) else {}
    message = _first_cf_message(body)
    error = body.get("error")
    if not message and isinstance(error, dict):
        message = str(error.get("message") or error.get("type") or error.get("code") or "")
    if not message and isinstance(error, str):
        message = error
    if not message:
        message = (raw.text_body or "")[:300].replace("\n", " ")
    return message.strip()


def is_billing_or_credit_block(detail: str) -> bool:
    """認証後に返る請求設定・残高不足を、キー不正と区別する。"""
    low = (detail or "").lower()
    return any(marker in low for marker in (
        "credit card",
        "no credits remaining",
        "credit balance",
        "insufficient credit",
        "insufficient_quota",
        "billing",
        "payment method",
    ))


def _cloudflare_token_probe(rep: DiagReport, api_key: str, *, verify_tls: bool = True) -> None:
    """Cloudflare API トークンを専用エンドポイントで検証し、結果を Auth / HTTP response
    項目へ反映する（api_key_dialog はこの2項目で保存可否を決める）。"""
    req = VlmHttpRequest(method="GET", url=_CLOUDFLARE_TOKEN_VERIFY_URL,
                         headers={"Authorization": f"Bearer {api_key}"})
    raw = execute_http(req, connect_timeout=10.0, read_timeout=15.0, verify_tls=verify_tls)
    if not isinstance(raw, RawHttpResponse):
        rep.add("HTTP response", DiagStatus.FAIL, f"{raw.reason.value}: {raw.message}")
        rep.add("Caption extraction", DiagStatus.SKIP, "no response to check")
        return
    rep.http_status = raw.status
    body = raw.json_body if isinstance(raw.json_body, dict) else {}
    result = body.get("result") if isinstance(body.get("result"), dict) else {}
    token_status = str(result.get("status", "")).lower()
    auth_item = rep.item("Auth")
    if raw.status == 200 and body.get("success") is True and token_status in ("", "active"):
        rep.add("HTTP response", DiagStatus.PASS, "token valid and active")
        if auth_item is not None:
            auth_item.status = DiagStatus.PASS
            auth_item.detail = "Cloudflare token verified"
    elif raw.status in (401, 403) or body.get("success") is False:
        msg = _first_cf_message(body) or f"{raw.status} token rejected"
        rep.add("HTTP response", DiagStatus.FAIL, msg)
        if auth_item is not None:
            auth_item.status = DiagStatus.FAIL
            auth_item.detail = msg
    elif raw.status == 200 and body.get("success") is True:
        rep.add("HTTP response", DiagStatus.FAIL, f"token is {token_status or 'not active'}")
        if auth_item is not None:
            auth_item.status = DiagStatus.FAIL
            auth_item.detail = f"token is {token_status or 'not active'}"
    else:
        rep.add("HTTP response", DiagStatus.WARN, f"HTTP {raw.status}")
    rep.add("Caption extraction", DiagStatus.SKIP, "Cloudflare token-verify check only")


def diagnose(conn: VlmConnection, api_key: str | None, *,
             do_live_request: bool = True) -> DiagReport:
    """接続を一括診断する。do_live_request=False なら実 HTTP を打たず静的検査だけ。"""
    rep = DiagReport(connection_id=conn.connection_id)

    # 1. URL / 設定値の形式
    parsed = urlparse(conn.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        rep.add("URL format", DiagStatus.FAIL, f"invalid base_url: {conn.base_url!r}")
        return rep
    is_cloudflare = (conn.provider_id == "cloudflare"
                     or (parsed.hostname or "").endswith("api.cloudflare.com"))
    # base_url にテンプレート変数（`{account_id}` 等）が残っていると、そのまま実
    # リクエストして意味不明な 404 になる。Cloudflare のアカウント ID 未設定だけは
    # WARN 止まり（キー自体は下の専用エンドポイントで検証できる）。それ以外の
    # 未展開変数は原因を明示して打ち切る。
    unresolved = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", conn.base_url)
    cf_missing_account = is_cloudflare and unresolved == ["{account_id}"]
    if unresolved and not cf_missing_account:
        rep.add("URL format", DiagStatus.FAIL,
                f"unresolved placeholder in base_url: {' '.join(unresolved)}")
        return rep
    if cf_missing_account:
        rep.add("URL format", DiagStatus.WARN,
                "Cloudflare account ID is not set - the key can still be verified, "
                "but this route will not run until Register API key is opened and the "
                "Account ID is entered there")
    elif parsed.scheme == "http" and not _looks_localish(parsed.hostname or ""):
        rep.add("URL format", DiagStatus.WARN, "plain http to a non-local host")
    else:
        rep.add("URL format", DiagStatus.PASS, conn.base_url)
    if not conn.model_id:
        rep.add("Model ID", DiagStatus.FAIL, "model_id is empty")
    else:
        rep.add("Model ID", DiagStatus.PASS, conn.model_id)
    if conn.protocol not in ("openai_chat_completions", "openai_responses",
                             "anthropic_messages", "gemini_generate_content"):
        rep.add("Protocol", DiagStatus.WARN, f"unknown protocol {conn.protocol!r}, treated as OpenAI compatible")
    else:
        rep.add("Protocol", DiagStatus.PASS, conn.protocol)

    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    if not do_live_request:
        # 静的検査モード: ネットワークに触れない（DNS / TCP / TLS / 実リクエストを飛ばす）。
        rep.add("DNS / TCP", DiagStatus.SKIP, "static check only")
        rep.add("TLS", DiagStatus.SKIP, "static check only")
    else:
        # 2. DNS / TCP
        try:
            socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
            rep.add("DNS / TCP", DiagStatus.PASS, f"{host}:{port}")
        except OSError as e:
            rep.add("DNS / TCP", DiagStatus.FAIL, f"cannot resolve/connect {host}:{port}: {e}")
            return rep

        # 3. TLS
        if parsed.scheme == "https":
            try:
                ctx = ssl.create_default_context()
                if not conn.verify_tls:
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                with socket.create_connection((host, port), timeout=conn.retry.connect_timeout_s) as sock:
                    with ctx.wrap_socket(sock, server_hostname=host):
                        pass
                rep.add("TLS", DiagStatus.PASS if conn.verify_tls else DiagStatus.WARN,
                        "verified" if conn.verify_tls else "verification disabled")
            except (ssl.SSLError, OSError) as e:
                rep.add("TLS", DiagStatus.FAIL, f"TLS handshake failed: {e}")
        else:
            rep.add("TLS", DiagStatus.SKIP, "plain http")

    # 4. Auth presence
    if conn.auth.type == "none":
        rep.add("Auth", DiagStatus.PASS, "no auth required")
    elif api_key:
        rep.add("Auth", DiagStatus.PASS, f"{conn.auth.type} credential present")
    else:
        rep.add("Auth", DiagStatus.FAIL, f"{conn.auth.type} required but no credential found")
        # 認証情報が無い状態で未認証リクエストを送ると、Vercel等の401本文だけが表示されて
        # 「キーが不正」と誤解しやすい。ネットワーク到達性は上で確認済みなので、ここで終了。
        rep.add("HTTP response", DiagStatus.SKIP, "credential missing")
        rep.add("Caption extraction", DiagStatus.SKIP, "credential missing")
        return rep

    # 4b. Account ID が未設定の Cloudflare だけは、専用エンドポイントでトークン単体を
    # 検証する。Account ID がある場合はこの先の chat/completions へ進み、アカウント・
    # Workers AI 権限・モデル・画像入力・応答抽出まで含めて接続を確認する。
    if cf_missing_account and conn.auth.type == "bearer" and api_key:
        if do_live_request:
            _cloudflare_token_probe(rep, api_key, verify_tls=conn.verify_tls)
        else:
            rep.add("HTTP response", DiagStatus.SKIP, "live request disabled")
            rep.add("Caption extraction", DiagStatus.SKIP, "live request disabled")
        return rep

    # 5. Request build
    try:
        prepared = prepare_image(_tiny_test_image_bytes(), ImagePreprocessConfig(max_long_edge=64))
        # 診断は「200 が返り、テキストが取り出せるか」の確認。長文生成を待つ必要はないので
        # 出力トークンを絞る（既定 1024 のままだと Gemma 等で timeout する）。ただし thinking
        # 対応モデルは思考でトークンを食うので、回答が数トークンは残るよう 128 にする。
        profile = GenerationProfile(max_output_tokens=128)
        call = VlmCallSpec(conn.model_id, build_system_prompt(profile), build_user_prompt(profile), prepared, profile)
        protocol = get_protocol(conn.protocol)
        if conn.text_path:
            protocol.default_text_path = conn.text_path
        req = protocol.build_request(conn.base_url, default_auth_key(conn.auth.type, api_key), call)
        apply_connection_auth(req, conn.auth.type, api_key, conn.auth.header_name, conn.auth.query_param)
        apply_request_headers(req, conn.request_headers)
        apply_request_body(req, conn.request_body)
        rep.add("Request build", DiagStatus.PASS, f"{req.method} {req.url}")
    except Exception as e:  # noqa: BLE001 - 診断なので全部拾う
        rep.add("Request build", DiagStatus.FAIL, f"{type(e).__name__}: {e}")
        return rep

    # 6. Image input format (静的確認のみ)
    rep.add("Image input", DiagStatus.PASS, f"{prepared.mime_type}, base64/data-url ready")

    if not do_live_request:
        rep.add("HTTP response", DiagStatus.SKIP, "live request disabled")
        rep.add("Caption extraction", DiagStatus.SKIP, "live request disabled")
        return rep

    # 7-11. 実リクエスト。診断は「疎通確認」なのでタイムアウトは短めに固定する
    # （設定ダイアログを閉じるときの待ち時間を抑える。実生成は本来の timeout を使う）。
    raw = execute_http(req, connect_timeout=min(conn.retry.connect_timeout_s, 10.0),
                       read_timeout=min(conn.retry.read_timeout_s, 30.0),
                       verify_tls=conn.verify_tls)
    if not isinstance(raw, RawHttpResponse):
        rep.add("HTTP response", DiagStatus.FAIL, f"{raw.reason.value}: {raw.message}")
        return rep

    rep.http_status = raw.status
    provider_detail = _response_error_detail(raw)
    billing_blocked = is_billing_or_credit_block(provider_detail)
    if raw.status == 200:
        rep.add("HTTP response", DiagStatus.PASS, "200 OK")
    elif billing_blocked:
        detail = f"{raw.status} billing / credits unavailable"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.FAIL, detail)
    elif raw.status in (401, 403):
        detail = f"{raw.status} auth rejected"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.FAIL, detail)
    elif raw.status in (404, 400, 422):
        detail = f"{raw.status} model / request rejected (auth was accepted)"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.FAIL, detail)
    elif raw.status == 429:
        detail = "429 rate limited (endpoint reachable)"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.WARN, detail)
    else:
        detail = f"HTTP {raw.status}"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.WARN, detail)

    # 4'. Auth の判定を実応答で上書きする。請求設定・残高不足が明記された403等は
    # 認証成功として扱い、それ以外の401/403だけを「キー不正」とする。
    auth_item = rep.item("Auth")
    if auth_item is not None and conn.auth.type != "none":
        if billing_blocked:
            auth_item.status = DiagStatus.PASS
            auth_item.detail = f"accepted; billing / credits unavailable (server responded {raw.status})"
        elif raw.status in (401, 403):
            auth_item.status = DiagStatus.FAIL
            auth_item.detail = f"rejected by the server ({raw.status})"
        else:
            auth_item.status = DiagStatus.PASS
            auth_item.detail = f"accepted (server responded {raw.status})"

    ext_status, ext_detail = _classify_extraction(raw, protocol)
    rep.add("Caption extraction", ext_status, ext_detail)

    # 10. Rate-limit headers（情報表示のみ。無くても正常＝多くの API は付けない。
    # その場合は 429 応答の Retry-After を見て事後クールダウンする。WARN にしない）。
    rl_names = [k for k in raw.headers if k.lower().startswith(("x-ratelimit", "ratelimit", "retry-after"))]
    rep.add("Rate-limit info", DiagStatus.PASS,
            ", ".join(rl_names) if rl_names else "none exposed (falls back to 429 Retry-After)")

    return rep


def _classify_extraction(raw: RawHttpResponse, protocol) -> tuple[DiagStatus, str]:
    """live レスポンスからテキストが取り出せるかを判定する。

    診断は出力トークンを絞るので、テキストが出る前に打ち切られること（finishReason=
    MAX_TOKENS / length）がある。その場合はエンドポイント・認証・リクエスト形状は通って
    いるので WARN 止まり。真に形が違うときだけ FAIL（本文の頭を付ける）。
    """
    parsed = protocol.parse_response(raw.status, raw.json_body, raw.text_body)
    if parsed.ok:
        return DiagStatus.PASS, f"got {len(parsed.text or '')} chars"
    if parsed.error and parsed.error.reason is VlmErrorReason.CONTENT_POLICY:
        return DiagStatus.WARN, "content policy on the test image (extraction path unverified)"
    if raw.status != 200:
        return DiagStatus.SKIP, "no successful response to extract from"
    finish_reason = str(
        extract_by_path(raw.json_body, "candidates[0].finishReason")
        or extract_by_path(raw.json_body, "choices[0].finish_reason")
        or extract_by_path(raw.json_body, "incomplete_details.reason")
        or extract_by_path(raw.json_body, "stop_reason") or ""
    ).upper()
    if finish_reason in ("MAX_TOKENS", "MAX_OUTPUT_TOKENS", "LENGTH"):
        return DiagStatus.WARN, "response truncated at the diagnostic token cap (endpoint reachable)"
    preview = (raw.text_body or "")[:200].replace("\n", " ")
    detail = "200 OK but the response text path did not match"
    if preview:
        detail += f" — body starts: {preview}"
    return DiagStatus.FAIL, detail


def _looks_localish(host: str) -> bool:
    h = host.lower()
    return h in ("localhost", "127.0.0.1", "::1") or h.startswith("192.168.") or h.startswith("10.") or h.endswith(".local")
