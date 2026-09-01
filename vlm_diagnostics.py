"""接続診断（260901_VLM_spec.md 12章 / design.md 4.6節）。

「その接続設定が実際に動くか」を一括で確認する。選択画像の品質を見る機能ではない。
結果は接続定義に書き戻さず、状態キャッシュとして扱う。設定変更で無効化する。
"""
from __future__ import annotations

import io
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
    VlmCallSpec, apply_connection_auth, default_auth_key, extract_by_path, get_protocol,
)
from vlm_transport import RawHttpResponse, execute_http


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


def _tiny_test_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def diagnose(conn: VlmConnection, api_key: str | None, *,
             do_live_request: bool = True) -> DiagReport:
    """接続を一括診断する。do_live_request=False なら実 HTTP を打たず静的検査だけ。"""
    rep = DiagReport(connection_id=conn.connection_id)

    # 1. URL / 設定値の形式
    parsed = urlparse(conn.base_url)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        if parsed.scheme == "http" and not _looks_localish(parsed.hostname or ""):
            rep.add("URL format", DiagStatus.WARN, "plain http to a non-local host")
        else:
            rep.add("URL format", DiagStatus.PASS, conn.base_url)
    else:
        rep.add("URL format", DiagStatus.FAIL, f"invalid base_url: {conn.base_url!r}")
        return rep
    if not conn.model_id:
        rep.add("Model ID", DiagStatus.FAIL, "model_id is empty")
    else:
        rep.add("Model ID", DiagStatus.PASS, conn.model_id)
    if conn.protocol not in ("openai_chat_completions", "gemini_generate_content"):
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
    if raw.status == 200:
        rep.add("HTTP response", DiagStatus.PASS, "200 OK")
    elif raw.status in (401, 403):
        rep.add("HTTP response", DiagStatus.FAIL, f"{raw.status} auth rejected")
    elif raw.status in (404, 400, 422):
        rep.add("HTTP response", DiagStatus.FAIL, f"{raw.status} model / request rejected (auth was accepted)")
    elif raw.status == 429:
        rep.add("HTTP response", DiagStatus.WARN, "429 rate limited (endpoint reachable)")
    else:
        rep.add("HTTP response", DiagStatus.WARN, f"HTTP {raw.status}")

    # 4'. Auth の判定を実応答で上書きする。401/403 だけが「キー不正」で、それ以外の
    # ステータスが返っているなら認証は通っている（モデル ID 違い等は別問題）。
    auth_item = rep.item("Auth")
    if auth_item is not None and conn.auth.type != "none":
        if raw.status in (401, 403):
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
        or extract_by_path(raw.json_body, "choices[0].finish_reason") or ""
    ).upper()
    if finish_reason in ("MAX_TOKENS", "LENGTH"):
        return DiagStatus.WARN, "response truncated at the diagnostic token cap (endpoint reachable)"
    preview = (raw.text_body or "")[:200].replace("\n", " ")
    detail = "200 OK but the response text path did not match"
    if preview:
        detail += f" — body starts: {preview}"
    return DiagStatus.FAIL, detail


def _looks_localish(host: str) -> bool:
    h = host.lower()
    return h in ("localhost", "127.0.0.1", "::1") or h.startswith("192.168.") or h.startswith("10.") or h.endswith(".local")
