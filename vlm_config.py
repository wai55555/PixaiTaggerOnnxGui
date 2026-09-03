"""AppSettings と vlm_connections.json から VLM 実行時オブジェクトを組み立てる橋渡し。

app_settings.py（既存設定の器）と vlm_* 群（機能本体）を疎結合に保つため、両者を知る
のはこのモジュールだけにする。260901_VLM_spec.md 16章 / design.md 5章・11章。
"""
from __future__ import annotations

import dataclasses
import json
import os
import uuid

from constants import BASE_DIR
from utils import write_debug_log
from vlm_connections import VlmConnection, default_builtin_connections
from vlm_models import (
    ModelBinding, ModelIdentityStatus, VlmModelProfile, default_registry,
)
from vlm_profiles import GenerationProfile
from vlm_router import RouterPolicy, parse_execution_mode

VLM_CONNECTIONS_PATH = BASE_DIR / "vlm_connections.json"
VLM_CONNECTIONS_VERSION = 1
VLM_PROFILES_PATH = BASE_DIR / "vlm_profiles.json"
VLM_PROFILES_VERSION = 1


# --- vlm_connections.json（カスタム接続定義） ---------------------------------------

def load_custom_connections() -> list[dict]:
    """vlm_connections.json のカスタム接続定義（生 dict）を返す。壊れていれば []。"""
    if not VLM_CONNECTIONS_PATH.is_file():
        return []
    try:
        with VLM_CONNECTIONS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        write_debug_log(f"vlm_config: cannot read {VLM_CONNECTIONS_PATH.name}: {e}")
        return []
    if not isinstance(data, dict):
        return []
    conns = data.get("connections")
    if not isinstance(conns, list):
        return []
    return [c for c in conns if isinstance(c, dict) and c.get("connection_id")]


def save_custom_connections(connections: list[dict]) -> bool:
    """カスタム接続定義を原子的に書き出す。秘密値は含めない前提（呼び出し側の責務）。"""
    payload = {"version": VLM_CONNECTIONS_VERSION, "connections": connections}
    tmp = VLM_CONNECTIONS_PATH.with_name(f"{VLM_CONNECTIONS_PATH.stem}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, VLM_CONNECTIONS_PATH)
        return True
    except (OSError, TypeError, ValueError) as e:
        write_debug_log(f"vlm_config: cannot write {VLM_CONNECTIONS_PATH.name}: {e}")
        return False
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def new_connection_id(prefix: str = "custom") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --- vlm_profiles.json（利用者が作る／編集するモデルプロファイル） -------------------
# 出荷プロファイルは推定なので、モデル一覧で見つけた実 ID を束ねた「自分のプロファイル」を
# ここに保存する。binding の identity は UNKNOWN（接続診断のフル PASS で verified 昇格）。

def load_user_profiles() -> list[dict]:
    if not VLM_PROFILES_PATH.is_file():
        return []
    try:
        with VLM_PROFILES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        write_debug_log(f"vlm_config: cannot read {VLM_PROFILES_PATH.name}: {e}")
        return []
    profs = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profs, list):
        return []
    return [p for p in profs if isinstance(p, dict) and p.get("profile_id")]


def save_user_profiles(profiles: list[dict]) -> bool:
    payload = {"version": VLM_PROFILES_VERSION, "profiles": profiles}
    tmp = VLM_PROFILES_PATH.with_name(f"{VLM_PROFILES_PATH.stem}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, VLM_PROFILES_PATH)
        return True
    except (OSError, TypeError, ValueError) as e:
        write_debug_log(f"vlm_config: cannot write {VLM_PROFILES_PATH.name}: {e}")
        return False
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _profile_from_dict(d: dict) -> VlmModelProfile | None:
    pid = str(d.get("profile_id", "")).strip()
    if not pid:
        return None
    bindings: dict[str, ModelBinding] = {}
    for prov, b in (d.get("bindings") or {}).items():
        if not isinstance(b, dict):
            continue
        mid = str(b.get("model_id", "")).strip()
        if not mid:
            continue
        bindings[prov] = ModelBinding(provider_id=prov, model_id=mid,
                                      identity_status=ModelIdentityStatus.UNKNOWN,
                                      free_route=bool(b.get("free_route", False)))
    if not bindings:
        return None
    return VlmModelProfile(
        profile_id=pid,
        display_name=str(d.get("display_name", pid)),
        canonical_model_id=str(d.get("canonical_model_id", pid)),
        family=str(d.get("family", "")),
        base_model=str(d.get("base_model", d.get("canonical_model_id", pid))),
        quantization="unknown",
        aliases=tuple(str(a) for a in (d.get("aliases") or []) if str(a).strip()),
        bindings=bindings,
    )


def user_profile_objects() -> list[VlmModelProfile]:
    return [p for p in (_profile_from_dict(d) for d in load_user_profiles()) if p is not None]


def all_profiles() -> list[VlmModelProfile]:
    """出荷プロファイル ＋ 利用者プロファイル（同じ profile_id なら利用者側で上書き）。"""
    shipped = default_registry().all_profiles()
    users = user_profile_objects()
    umap = {p.profile_id: p for p in users}
    out = [umap.get(p.profile_id, p) for p in shipped]
    seen = {p.profile_id for p in shipped}
    out.extend(p for p in users if p.profile_id not in seen)
    return out


def is_user_profile(profile_id: str) -> bool:
    return any(d.get("profile_id") == profile_id for d in load_user_profiles())


def new_profile_id(display_name: str = "") -> str:
    base = "".join(c if c.isalnum() else "-" for c in display_name.lower()).strip("-")
    return f"user-{base[:24] or 'profile'}-{uuid.uuid4().hex[:6]}"


# --- 実行時オブジェクトの組み立て -------------------------------------------------

def build_connection_map(vlm_settings, model_profile=None) -> dict[str, VlmConnection]:
    """内蔵接続 + vlm_connections.json のカスタム接続を connection_id 辞書で返す。

    `model_profile` を渡すと、内蔵接続の `model_id` をそのプロファイルの binding から埋める
    （`[Vlm] model_id_overrides` の `<profile>:<provider>=<id>` が最優先）。その provider に
    binding が無ければ「このモデルを提供しない経路」として無効化する。None のときは
    テンプレートの既定 model_id をそのまま使う（旧挙動）。APIキーは vlm_secrets が別途
    secret_ref から解決するのでここでは埋めない。
    """
    paid_set = {p.strip() for p in str(getattr(vlm_settings, "paid_connections", "")).split(",") if p.strip()}
    cf_account = (str(getattr(vlm_settings, "cloudflare_account_id", "") or "").strip()
                  or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip())
    anthropic_workspace = (
        str(getattr(vlm_settings, "anthropic_workspace_id", "") or "").strip()
        or os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip())
    overrides = vlm_settings.model_id_override_map() if hasattr(vlm_settings, "model_id_override_map") else {}
    profile_id = getattr(model_profile, "profile_id", "") or getattr(vlm_settings, "model_profile_id", "")
    result: dict[str, VlmConnection] = {}
    for conn in default_builtin_connections():
        # 有料継続の個別許可を反映（provider_id 単位）。順序反映は Router 側で行う。
        if conn.provider_id in paid_set:
            conn.paid_continuation_allowed = True

        binding = model_profile.binding_for(conn.provider_id) if model_profile is not None else None
        override = overrides.get(f"{profile_id}:{conn.provider_id}")
        if override:
            conn.model_id = override
            if binding is not None:
                conn.is_known_free_route = binding.free_route
        elif binding is not None:
            conn.model_id = binding.model_id or conn.model_id
            conn.is_known_free_route = binding.free_route
        elif model_profile is not None:
            # 選択プロファイルはこの provider を経路に持たない。
            conn.enabled = False
        if not conn.model_id:
            conn.enabled = False

        # Cloudflare はアカウント ID を URL に埋める。未設定なら無効化しておく
        # （build_request で `{account_id}` のままリクエストしても必ず失敗するため）。
        if "{account_id}" in conn.base_url:
            if cf_account:
                conn.base_url = conn.base_url.replace("{account_id}", cf_account)
            else:
                conn.enabled = False
        if conn.provider_id == "anthropic" and anthropic_workspace:
            conn.request_headers["anthropic-workspace-id"] = anthropic_workspace
        result[conn.connection_id] = conn
    for raw in load_custom_connections():
        try:
            conn = VlmConnection.from_mapping(raw)
        except (KeyError, ValueError) as e:
            write_debug_log(f"vlm_config: skip malformed custom connection: {e}")
            continue
        result[conn.connection_id] = conn
    return result


def build_generation_profile(vlm_settings) -> GenerationProfile:
    """AppSettings.vlm の生成関連フィールドから GenerationProfile を作る。"""
    return GenerationProfile.from_mapping({
        "profile_id": vlm_settings.generation_profile_id,
        "language": vlm_settings.language,
        "detail_level": vlm_settings.detail_level,
        "sentence_mode": vlm_settings.sentence_mode,
        "character_name_mode": vlm_settings.character_name_mode,
        "markdown": vlm_settings.markdown,
        "max_output_tokens": vlm_settings.max_output_tokens,
        "image_max_long_edge": vlm_settings.image_max_long_edge,
    })


def build_router_policy(vlm_settings) -> RouterPolicy:
    return RouterPolicy(
        execution_mode=parse_execution_mode(vlm_settings.execution_mode),
        free_only=bool(vlm_settings.free_only),
        paid_continuation=bool(vlm_settings.paid_continuation),
        selected_connection_id=(vlm_settings.selected_connection_id or None),
        allow_declared_identity=not bool(getattr(vlm_settings, "strict_identity", False)),
    )


def resolve_model_profile(vlm_settings):
    by_id = {p.profile_id: p for p in all_profiles()}
    profile = by_id.get(vlm_settings.model_profile_id)
    if profile is None:
        profile = default_registry().resolve_alias(vlm_settings.model_profile_id)
    return _apply_verified_promotions(profile, vlm_settings) if profile is not None else None


def _binding_token(profile_id: str, provider_id: str) -> str:
    return f"{profile_id}:{provider_id}"


def _apply_verified_promotions(profile, vlm_settings):
    """`[Vlm] verified_bindings` に載っている binding を VERIFIED へ引き上げる。

    出荷時 identity は控えめ（多くが DECLARED）なので、接続診断のフル PASS や 1枚テスト
    成功で「実際に期待どおり動いた」と分かった binding をここで昇格させる。UNKNOWN の
    ままにはしない（実証済みなので）。既に VERIFIED のものはそのまま。
    """
    verified = vlm_settings.verified_set()
    if not verified:
        return profile
    changed = {}
    for pid, binding in profile.bindings.items():
        if (_binding_token(profile.profile_id, pid) in verified
                and binding.identity_status is not ModelIdentityStatus.VERIFIED):
            changed[pid] = dataclasses.replace(
                binding, identity_status=ModelIdentityStatus.VERIFIED, provider_constraint=None)
        else:
            changed[pid] = binding
    return dataclasses.replace(profile, bindings=changed)


def set_model_id_override(vlm_settings, provider_id: str, model_id: str,
                          *, profile_id: str | None = None) -> None:
    """内蔵経路のモデル ID を上書きする。空文字なら上書きを解除。

    呼び出し側で `save_config(settings)` を実行して永続化すること。
    """
    key = f"{profile_id or vlm_settings.model_profile_id}:{provider_id}"
    m = vlm_settings.model_id_override_map()
    model_id = (model_id or "").strip()
    if model_id:
        m[key] = model_id
    else:
        m.pop(key, None)
    vlm_settings.model_id_overrides = ",".join(f"{k}={v}" for k, v in sorted(m.items()))


def mark_binding_verified(vlm_settings, provider_id: str, *, profile_id: str | None = None) -> bool:
    """接続診断／1枚テストが成功したときに呼ぶ。既に載っていれば False。

    呼び出し側で `save_config(settings)` を実行して永続化すること。
    """
    if not provider_id:
        return False
    token = _binding_token(profile_id or vlm_settings.model_profile_id, provider_id)
    current = vlm_settings.verified_set()
    if token in current:
        return False
    current.add(token)
    vlm_settings.verified_bindings = ",".join(sorted(current))
    return True


KNOWN_BUILTIN_PROVIDERS = (
    "gemini", "openrouter", "cloudflare", "groq", "nvidia", "mistral",
    "huggingface", "vercel", "openai", "anthropic",
    # "ovhcloud",  # 日本居住者環境で実機検証できるまで無効
)


def ordered_builtin_provider_ids(vlm_settings, model_profile=None) -> list[str]:
    """config の connection_order を「順序 かつ 有効集合」として解釈する。

    設定ダイアログのフォールバック経路リストでチェックを外した provider は
    connection_order から除かれるので足し戻さない（「無効化」を尊重）。ただし
    `model_profile` は引数互換のため受け取るが、binding にあるだけの provider は
    足し戻さない。UI で明示的にチェックした経路だけを実行対象にする。
    設定が空のときだけ既定の3つに戻す。
    """
    known = set(KNOWN_BUILTIN_PROVIDERS)
    seen: set[str] = set()
    out: list[str] = []
    for p in vlm_settings.order_list():
        if p in known and p not in seen:
            seen.add(p)
            out.append(p)
    return out or ["gemini", "openrouter", "cloudflare"]
