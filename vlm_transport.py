"""HTTP 実行層とフォールバック実行ループ（260901_VLM_spec.md 7・8章 / design.md 4.9節）。

`vlm_protocols` が作った VlmHttpRequest を requests で実行し、結果を
`VlmProtocol.parse_response` へ渡す。ここに「どの接続を、どの順で、どうリトライするか」
の実行ループ（VlmExecutor）も置く。UI スレッドからは呼ばない。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from utils import write_debug_log
from vlm_connections import VlmConnection
from vlm_errors import VlmAttemptError, VlmErrorClass, VlmErrorReason
from vlm_image import PreparedImage
from vlm_profiles import GenerationProfile
from vlm_protocols import (
    VlmCallSpec, VlmParseResult, apply_connection_auth, apply_request_body, apply_request_headers,
    default_auth_key, get_protocol,
)
from vlm_ratelimit import RateLimitState, update_from_429

StopChecker = Callable[[], bool]


@dataclass(frozen=True)
class RawHttpResponse:
    status: int
    headers: dict[str, str]
    json_body: Any
    text_body: str


def execute_http(req, *, connect_timeout: float, read_timeout: float,
                 verify_tls: bool = True) -> RawHttpResponse | VlmAttemptError:
    """1回の HTTP リクエストを実行する。ネットワーク例外は VlmAttemptError にして返す。"""
    try:
        resp = requests.request(
            req.method, req.url,
            headers=req.headers or None,
            params=req.params or None,
            json=req.json_body or None,
            timeout=(connect_timeout, read_timeout),
            verify=verify_tls,
        )
    except requests.exceptions.Timeout:
        return VlmAttemptError(VlmErrorReason.TIMEOUT, None, "request timed out")
    except requests.exceptions.SSLError as e:
        return VlmAttemptError(VlmErrorReason.NETWORK, None, f"TLS error: {e}")
    except requests.exceptions.ConnectionError as e:
        return VlmAttemptError(VlmErrorReason.NETWORK, None, f"connection error: {e}")
    except requests.exceptions.RequestException as e:
        return VlmAttemptError(VlmErrorReason.UNKNOWN, None, f"request failed: {e}")

    text_body = resp.text or ""
    body: Any = None
    ctype = resp.headers.get("Content-Type", "")
    if "json" in ctype.lower() or (text_body[:1] in ("{", "[")):
        try:
            body = resp.json()
        except (ValueError, json.JSONDecodeError):
            body = None
    return RawHttpResponse(status=resp.status_code, headers=dict(resp.headers),
                           json_body=body, text_body=text_body)


@dataclass
class ConnectionRuntime:
    """1ジョブの実行中に接続ごとに変わる状態（スナップショットではない）。"""
    excluded_reason: str = ""            # 空なら候補として生きている
    consecutive_timeouts: int = 0
    retried_same_this_image: bool = False
    rate_limit: RateLimitState | None = None

    @property
    def is_excluded(self) -> bool:
        return bool(self.excluded_reason)


@dataclass
class AttemptRecord:
    connection_id: str
    model_id: str
    http_status: int | None
    error_reason: str
    error_class: str
    at_utc: float


@dataclass
class ImageResult:
    text: str | None = None
    connection_id: str = ""
    model_id: str = ""
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    error: VlmAttemptError | None = None
    stopped: bool = False
    # プロンプト形式エラーなど、続行しても全画像で同じ失敗になる致命的エラー。
    # 呼び出し側（Worker）はこれを見てバッチを打ち切る（spec.md 8.1 stop_job）。
    stop_job: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.stopped and bool((self.text or "").strip())


class VlmExecutor:
    """フォールバック順に接続を試し、最初に成功したキャプションを返す。

    接続ごとの ConnectionRuntime を持ち、EXCLUDE / cooldown を次画像へ引き継ぐ。
    別モデルへの切り替えは行わない（候補は全て同一モデルプロファイル）。
    """

    def __init__(self, connections: dict[str, VlmConnection],
                 secret_resolver: Callable[[str], str | None],
                 *, stop_checker: StopChecker | None = None):
        self._connections = connections
        self._resolve_secret = secret_resolver
        self._stop = stop_checker or (lambda: False)
        self._runtime: dict[str, ConnectionRuntime] = {}

    def runtime(self, cid: str) -> ConnectionRuntime:
        return self._runtime.setdefault(cid, ConnectionRuntime())

    def live_candidates(self, ordered_ids: list[str], now: float | None = None) -> list[str]:
        now = now if now is not None else time.time()
        out = []
        for cid in ordered_ids:
            rt = self.runtime(cid)
            if rt.is_excluded:
                continue
            if rt.rate_limit is not None and rt.rate_limit.in_cooldown(now):
                continue
            out.append(cid)
        return out

    def caption_one(self, spec_base: dict, ordered_connection_ids: list[str]) -> ImageResult:
        """1画像に対しフォールバック順で試行する。

        spec_base: {"image": PreparedImage, "profile": GenerationProfile,
                    "system_prompt": str, "user_prompt": str}
        """
        result = ImageResult()
        image: PreparedImage = spec_base["image"]
        profile: GenerationProfile = spec_base["profile"]

        for cid in list(ordered_connection_ids):
            if self._stop():
                result.stopped = True
                return result
            rt = self.runtime(cid)
            if rt.is_excluded:
                continue
            if rt.rate_limit is not None and rt.rate_limit.in_cooldown():
                continue
            conn = self._connections.get(cid)
            if conn is None:
                rt.excluded_reason = "missing_connection"
                continue

            rt.retried_same_this_image = False
            outcome = self._try_connection(conn, image, profile, spec_base, result)
            if outcome == "success":
                return result
            if outcome == "stopped":
                result.stopped = True
                return result
            if outcome == "stop_job":
                # プロンプト形式エラー等。呼び出し側でジョブ停止扱いにする。
                result.stop_job = True
                return result
            # failover / exclude はループ継続

        if result.error is None:
            result.error = VlmAttemptError(VlmErrorReason.UNKNOWN, None, "all candidates exhausted")
        return result

    def _try_connection(self, conn: VlmConnection, image: PreparedImage,
                        profile: GenerationProfile, spec_base: dict, result: ImageResult) -> str:
        protocol = get_protocol(conn.protocol)
        # カスタム接続のレスポンス抽出パス上書き（fresh なインスタンスなので安全）。
        if conn.text_path:
            protocol.default_text_path = conn.text_path
        api_key = self._resolve_secret(conn.auth.secret_ref) if conn.auth.type != "none" else None
        default_key = default_auth_key(conn.auth.type, api_key)
        call = VlmCallSpec(
            model_id=conn.model_id,
            system_prompt=spec_base["system_prompt"],
            user_prompt=spec_base["user_prompt"],
            image=image,
            profile=profile,
        )
        rt = self.runtime(conn.connection_id)
        # 同一接続での再試行回数の上限。分類ロジックが必ず有限回で FAILOVER へ倒す
        # 想定だが、将来の分類変更で無限ループにならないための保険も兼ねる。
        same_conn_attempts = 0
        max_same_conn_attempts = max(1, conn.retry.retry_same_max) + 1

        while True:
            if self._stop():
                return "stopped"
            same_conn_attempts += 1
            req = protocol.build_request(conn.base_url, default_key, call)
            apply_connection_auth(req, conn.auth.type, api_key,
                                  conn.auth.header_name, conn.auth.query_param)
            apply_request_headers(req, conn.request_headers)
            apply_request_body(req, conn.request_body)
            raw = execute_http(req, connect_timeout=conn.retry.connect_timeout_s,
                               read_timeout=conn.retry.read_timeout_s,
                               verify_tls=conn.verify_tls)
            if isinstance(raw, VlmAttemptError):
                parsed = VlmParseResult(error=raw)
            else:
                if raw.status == 429:
                    rt.rate_limit = update_from_429(rt.rate_limit or RateLimitState(conn.connection_id), raw.headers)
                parsed = protocol.parse_response(raw.status, raw.json_body, raw.text_body)

            if parsed.ok:
                rt.consecutive_timeouts = 0
                result.error = None          # 先行接続の失敗を引きずらない
                result.text = parsed.text
                result.connection_id = conn.connection_id
                result.model_id = conn.model_id
                result.finish_reason = parsed.finish_reason
                result.prompt_tokens = parsed.prompt_tokens
                result.completion_tokens = parsed.completion_tokens
                return "success"

            err = parsed.error or VlmAttemptError(VlmErrorReason.BAD_RESPONSE, None, "no text and no error")
            if err.reason is VlmErrorReason.TIMEOUT:
                rt.consecutive_timeouts += 1
            else:
                rt.consecutive_timeouts = 0

            cls = err.classify(consecutive_timeouts=rt.consecutive_timeouts,
                               already_retried_same=rt.retried_same_this_image)
            result.attempts.append(AttemptRecord(
                connection_id=conn.connection_id, model_id=conn.model_id,
                http_status=err.http_status, error_reason=err.reason.value,
                error_class=cls.value, at_utc=time.time()))
            result.error = err
            write_debug_log(
                f"vlm: {conn.connection_id} attempt -> {err.reason.value} "
                f"(http={err.http_status}) => {cls.value}")

            if cls is VlmErrorClass.RETRY_SAME and same_conn_attempts < max_same_conn_attempts:
                rt.retried_same_this_image = True
                continue
            if cls is VlmErrorClass.EXCLUDE:
                rt.excluded_reason = err.reason.value
                return "failover"
            if cls is VlmErrorClass.STOP_JOB:
                return "stop_job"
            # FAILOVER / FAIL_IMAGE(=最終候補で処理) は次候補へ
            return "failover"
