"""接続・モデル単位のレート制限／クールダウン状態（260901_VLM_spec.md 9章）。

リセット情報が明示されない場合は推測値として扱い、「サービス停止」とは表示しない。
推測クールダウンには上限を設け、次回の接続診断で再確認できるようにする。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

# 推測クールダウンの上限（秒）。明示リセットが無い 429 でここまで自動で待つ。
ESTIMATED_COOLDOWN_CAP_S = 15 * 60
# 明示情報の無い 429 の初期推測クールダウン。
ESTIMATED_COOLDOWN_DEFAULT_S = 60


@dataclass
class RateLimitState:
    connection_id: str
    last_429_at_utc: float | None = None
    retry_after_s: float | None = None
    reset_at_utc: float | None = None
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    cooldown_until_utc: float = 0.0
    source: str = ""  # header | response | estimated

    def in_cooldown(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return self.cooldown_until_utc > now

    def remaining_s(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, self.cooldown_until_utc - now)


def _parse_retry_after(value: str, now: float) -> float | None:
    """Retry-After: 秒数 or HTTP-date。"""
    value = (value or "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        dt = parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - now)
    except (TypeError, ValueError):
        return None


def _first_int(headers: dict, names: tuple[str, ...]) -> int | None:
    lower = {k.lower(): v for k, v in headers.items()}
    for n in names:
        v = lower.get(n.lower())
        if v is None:
            continue
        try:
            return int(float(v))
        except (TypeError, ValueError):
            continue
    return None


def update_from_429(state: RateLimitState, headers: dict, now: float | None = None) -> RateLimitState:
    """429 レスポンスのヘッダーからクールダウンを更新する。"""
    now = now if now is not None else time.time()
    lower = {k.lower(): v for k, v in (headers or {}).items()}
    state.last_429_at_utc = now

    retry_after = _parse_retry_after(lower.get("retry-after", ""), now)
    reset_epoch = None
    for name in ("x-ratelimit-reset", "x-ratelimit-reset-requests", "ratelimit-reset"):
        raw = lower.get(name)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        # 大きい値は epoch 秒、小さい値は「あと N 秒」とみなす。
        reset_epoch = val if val > 1e6 else now + val
        break

    state.rpm = _first_int(headers or {}, ("x-ratelimit-limit-requests", "x-ratelimit-limit", "ratelimit-limit"))
    state.tpm = _first_int(headers or {}, ("x-ratelimit-limit-tokens",))

    if retry_after is not None:
        state.retry_after_s = retry_after
        state.cooldown_until_utc = now + min(retry_after, ESTIMATED_COOLDOWN_CAP_S)
        state.source = "header"
    elif reset_epoch is not None:
        state.reset_at_utc = reset_epoch
        state.cooldown_until_utc = min(reset_epoch, now + ESTIMATED_COOLDOWN_CAP_S)
        state.source = "header"
    else:
        state.cooldown_until_utc = now + ESTIMATED_COOLDOWN_DEFAULT_S
        state.source = "estimated"
    return state


def clear(state: RateLimitState) -> RateLimitState:
    state.cooldown_until_utc = 0.0
    state.source = ""
    return state
