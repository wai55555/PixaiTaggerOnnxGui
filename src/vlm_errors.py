"""VLM リクエスト結果のエラー分類（260901_VLM_spec.md 8章 / implement_plan 10.2節）。

「どのエラーで同一接続リトライ / 次の接続へ / 除外 / 画像失敗 / ジョブ停止 とするか」を
一箇所に集約する。判断はここだけで行い、Worker / Router には分岐を持ち込まない。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VlmErrorClass(str, Enum):
    RETRY_SAME = "retry_same"      # 同一接続で再試行
    FAILOVER = "failover"          # 次の同一モデル接続へ
    EXCLUDE = "exclude"            # 今回の処理からこの接続を除外
    FAIL_IMAGE = "fail_image"      # この画像を失敗として残す
    STOP_JOB = "stop_job"          # ジョブ全体を停止（設定ミスの可能性）


class VlmErrorReason(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"          # HTTP 429
    SERVER_ERROR = "server_error"          # HTTP 5xx
    AUTH_ERROR = "auth_error"              # 401 / 403（ポリシー拒否を除く）
    MODEL_UNSUPPORTED = "model_unsupported"
    IMAGE_FORMAT_ERROR = "image_format_error"
    PROMPT_FORMAT_ERROR = "prompt_format_error"
    CONTENT_POLICY = "content_policy"
    EMPTY_RESPONSE = "empty_response"
    OUTPUT_LIMIT = "output_limit"
    BAD_RESPONSE = "bad_response"          # JSON 不正 / 抽出パス不一致
    NETWORK = "network"                    # DNS / TCP / TLS
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VlmAttemptError:
    reason: VlmErrorReason
    http_status: int | None = None
    message: str = ""
    # サーバーが返したエラーコード文字列（あれば）。ログ用。
    provider_code: str = ""

    def classify(self, *, consecutive_timeouts: int = 0,
                 already_retried_same: bool = False,
                 same_retries: int | None = None,
                 retry_same_max: int = 1,
                 retry_5xx: bool = True) -> VlmErrorClass:
        """このエラーに対する行動を返す（spec.md 8.2 の表）。

        - consecutive_timeouts: この接続で連続何回目のタイムアウトか（1 が初回）
        - already_retried_same: 同一接続でのリトライを今回すでに1回使ったか
        """
        r = self.reason
        retries = (int(same_retries) if same_retries is not None
                   else (1 if already_retried_same else 0))
        can_retry_same = max(0, int(retry_same_max)) > retries
        if r is VlmErrorReason.TIMEOUT:
            if can_retry_same:
                return VlmErrorClass.RETRY_SAME
            return VlmErrorClass.FAILOVER
        if r is VlmErrorReason.RATE_LIMITED:
            return VlmErrorClass.FAILOVER          # 待機しない
        if r is VlmErrorReason.SERVER_ERROR:
            return (VlmErrorClass.RETRY_SAME
                    if retry_5xx and can_retry_same else VlmErrorClass.FAILOVER)
        if r in (VlmErrorReason.AUTH_ERROR, VlmErrorReason.MODEL_UNSUPPORTED):
            return VlmErrorClass.EXCLUDE
        if r is VlmErrorReason.CONTENT_POLICY:
            return VlmErrorClass.FAILOVER
        if r is VlmErrorReason.EMPTY_RESPONSE:
            return VlmErrorClass.RETRY_SAME if can_retry_same else VlmErrorClass.FAILOVER
        if r is VlmErrorReason.OUTPUT_LIMIT:
            # 設定値を変えない限り同じ応答になるため、同一接続では無駄に再試行しない。
            return VlmErrorClass.FAILOVER
        if r is VlmErrorReason.IMAGE_FORMAT_ERROR:
            # 呼び出し側が画像を作り直せたら retry、無理なら failover。ここでは failover を既定に。
            return VlmErrorClass.FAILOVER
        if r is VlmErrorReason.PROMPT_FORMAT_ERROR:
            return VlmErrorClass.STOP_JOB
        if r in (VlmErrorReason.BAD_RESPONSE, VlmErrorReason.NETWORK, VlmErrorReason.UNKNOWN):
            return VlmErrorClass.FAILOVER
        return VlmErrorClass.FAILOVER


def _looks_like_prompt_format(message: str, provider_code: str = "") -> bool:
    low = f"{provider_code} {message}".lower()
    return any(marker in low for marker in (
        "messages[", "content must be", "content[", "image_url", "input_image",
        "inline_data", "inlineimage", "multimodal", "prompt format",
        "request body", "invalid image", "image input", "unsupported content",
    ))


def reason_from_http_status(status: int, message: str = "",
                            provider_code: str = "") -> VlmErrorReason:
    if status == 408:
        return VlmErrorReason.TIMEOUT
    if status == 429:
        return VlmErrorReason.RATE_LIMITED
    if status in (401, 403):
        return VlmErrorReason.AUTH_ERROR
    if status == 404:
        return VlmErrorReason.MODEL_UNSUPPORTED
    if status == 400 and _looks_like_prompt_format(message, provider_code):
        return VlmErrorReason.PROMPT_FORMAT_ERROR
    if 500 <= status <= 599:
        return VlmErrorReason.SERVER_ERROR
    if 400 <= status <= 499:
        return VlmErrorReason.BAD_RESPONSE
    if status == 200:
        # 200 なのにここへ来る = 本文が JSON でない／抽出パスに合致しない（本文が壊れている）。
        return VlmErrorReason.BAD_RESPONSE
    return VlmErrorReason.UNKNOWN
