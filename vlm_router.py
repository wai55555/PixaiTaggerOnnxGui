"""同一モデルの接続候補選定とフォールバック順の決定（260901_VLM_spec.md 7・14.4章）。

ここが決めるのは「どの接続を、どの順で試すか」まで。実際のリトライ実行ループは
vlm_worker 側に置く（Phase 5）。別モデルへの自動切り替えは絶対に行わない。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from vlm_connections import ConnectionKind, VlmConnection
from vlm_models import ModelIdentityStatus, VlmModelProfile


class ExecutionMode(str, Enum):
    BUILTIN_FALLBACK = "builtin_fallback"
    CUSTOM_SINGLE = "custom_single"


def parse_execution_mode(raw: object) -> ExecutionMode:
    try:
        return ExecutionMode(str(raw).strip().lower())
    except (ValueError, AttributeError):
        return ExecutionMode.BUILTIN_FALLBACK


@dataclass
class RouterPolicy:
    execution_mode: ExecutionMode = ExecutionMode.BUILTIN_FALLBACK
    free_only: bool = True
    paid_continuation: bool = False
    selected_connection_id: str | None = None
    # 「サポート済みのVLMサービスを使う」経路で、identity が DECLARED（同一と宣言済みだが
    # 実測未確認）の binding も候補に含めるか。どのサービスをどの順で試すかは利用者が
    # 経路リストで決めるため、既定は True。UNKNOWN は常に除外する。
    allow_declared_identity: bool = True


@dataclass
class CandidateSet:
    """選定結果。connection_ids が空なら理由が rejected_reason に入る。"""
    connection_ids: list[str] = field(default_factory=list)
    # connection_id -> 除外理由（UI 表示・ログ用）
    excluded: dict[str, str] = field(default_factory=dict)
    rejected_reason: str = ""

    @property
    def has_candidates(self) -> bool:
        return bool(self.connection_ids)


def select_candidates(
    profile: VlmModelProfile,
    connections: dict[str, VlmConnection],
    policy: RouterPolicy,
    *,
    cooldown_until: dict[str, float] | None = None,
    has_auth: dict[str, bool] | None = None,
    supports_image: dict[str, bool] | None = None,
    now: float | None = None,
) -> CandidateSet:
    """処理開始時に一度だけ呼ぶ。結果はジョブスナップショットへ固定する。"""
    now = now if now is not None else time.time()
    cooldown_until = cooldown_until or {}
    has_auth = has_auth or {}
    supports_image = supports_image or {}
    result = CandidateSet()

    if policy.execution_mode is ExecutionMode.CUSTOM_SINGLE:
        return _select_custom_single(connections, policy, cooldown_until, has_auth, now)

    # --- builtin_fallback ---
    # プロファイルの bindings 挿入順 = 既定の優先順位。UI で並べ替えた順は
    # connections 側の順序で表現される想定なので、両方を尊重して交差を取る。
    ordered_provider_ids = [pid for pid in profile.bindings if _provider_has_connection(pid, connections)]
    for provider_id in ordered_provider_ids:
        conn = _builtin_connection_for(provider_id, connections)
        if conn is None:
            continue
        cid = conn.connection_id
        binding = profile.binding_for(provider_id)

        if not conn.enabled:
            result.excluded[cid] = "disabled"
            continue
        if binding is None:
            result.excluded[cid] = "not_verified"
            continue
        status = binding.effective_identity_status()
        if status is ModelIdentityStatus.UNKNOWN:
            result.excluded[cid] = "identity_unknown"
            continue
        if status is not ModelIdentityStatus.VERIFIED and not policy.allow_declared_identity:
            result.excluded[cid] = "not_verified"
            continue
        if not has_auth.get(cid, False):
            result.excluded[cid] = "no_auth"
            continue
        if not supports_image.get(cid, True):
            result.excluded[cid] = "no_image_support"
            continue
        if cooldown_until.get(cid, 0.0) > now:
            result.excluded[cid] = "cooldown"
            continue
        if not _passes_fee_policy(conn, policy):
            result.excluded[cid] = "fee_policy"
            continue
        result.connection_ids.append(cid)

    if not result.connection_ids:
        # 量子化不明で厳格判定できないケースは、そもそも VERIFIED にならない前提。
        result.rejected_reason = "no_verified_free_candidate" if policy.free_only else "no_verified_candidate"
    return result


def _select_custom_single(connections, policy, cooldown_until, has_auth, now) -> CandidateSet:
    result = CandidateSet()
    cid = policy.selected_connection_id
    conn = connections.get(cid) if cid else None
    if conn is None:
        result.rejected_reason = "custom_connection_not_found"
        return result
    if not conn.is_custom:
        result.rejected_reason = "selected_connection_is_not_custom"
        return result
    if not conn.enabled:
        result.excluded[cid] = "disabled"
        result.rejected_reason = "custom_connection_disabled"
        return result
    if policy.free_only and conn.kind is ConnectionKind.CUSTOM_EXTERNAL:
        # 外部カスタムは料金を判定できないため free_only では実行不可（設定エラー表示）。
        result.excluded[cid] = "free_only_blocks_external_custom"
        result.rejected_reason = "free_only_blocks_external_custom"
        return result
    if conn.auth.type != "none" and not has_auth.get(cid, False):
        result.excluded[cid] = "no_auth"
        result.rejected_reason = "custom_connection_no_auth"
        return result
    if cooldown_until.get(cid, 0.0) > now:
        result.excluded[cid] = "cooldown"
        result.rejected_reason = "custom_connection_cooldown"
        return result
    result.connection_ids.append(cid)
    return result


def _passes_fee_policy(conn: VlmConnection, policy: RouterPolicy) -> bool:
    if not policy.free_only:
        return True
    if conn.free_for_automation:
        return True
    # 無料経路でなくても、有料継続が有効かつこの接続が個別許可されていれば候補に残す。
    return policy.paid_continuation and conn.paid_continuation_allowed


def _provider_has_connection(provider_id: str, connections: dict[str, VlmConnection]) -> bool:
    return _builtin_connection_for(provider_id, connections) is not None


def _builtin_connection_for(provider_id: str, connections: dict[str, VlmConnection]) -> VlmConnection | None:
    for conn in connections.values():
        if conn.kind is ConnectionKind.BUILTIN and conn.provider_id == provider_id:
            return conn
    return None
