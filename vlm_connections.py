"""内蔵接続とカスタム接続の定義（260901_VLM_spec.md 2.2・14章 / design.md 4.2・5.2節）。

- 内蔵接続: アプリが URL・プロトコル・既知の無料経路情報を持つ
  （Gemini / OpenRouter / Cloudflare / Groq / NVIDIA / Mistral / Hugging Face）
- カスタム接続: 利用者が登録する外部 API / ローカル VLM。同一モデル判定は行わない。
  外部・ローカルの判定は安全側（不明なら外部扱い）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ConnectionKind(str, Enum):
    BUILTIN = "builtin"
    CUSTOM_EXTERNAL = "custom_external"
    CUSTOM_LOCAL = "custom_local"


class ConnectionLocality(str, Enum):
    """カスタム接続編集画面での「接続先」選択。AUTO は URL から推測。"""
    AUTO = "auto"
    LOCAL = "local"
    EXTERNAL = "external"


_LOCAL_HOST_HINTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")


def _looks_local(base_url: str) -> bool:
    low = (base_url or "").lower()
    if not low:
        return False
    return any(h in low for h in _LOCAL_HOST_HINTS) or low.startswith("http://192.168.") \
        or low.startswith("http://10.") or ".local" in low


def resolve_custom_kind(locality: ConnectionLocality, base_url: str) -> ConnectionKind:
    """接続先種別を確定する。AUTO かつローカルと判定できなければ外部扱い（安全側）。"""
    if locality is ConnectionLocality.LOCAL:
        return ConnectionKind.CUSTOM_LOCAL
    if locality is ConnectionLocality.EXTERNAL:
        return ConnectionKind.CUSTOM_EXTERNAL
    return ConnectionKind.CUSTOM_LOCAL if _looks_local(base_url) else ConnectionKind.CUSTOM_EXTERNAL


@dataclass
class RetryPolicy:
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0
    retry_same_max: int = 1
    retry_5xx: bool = True
    use_retry_after_on_429: bool = False   # spec: 429 は待たず failover

    @classmethod
    def from_mapping(cls, data: dict | None) -> "RetryPolicy":
        d = data or {}
        base = cls()
        return cls(
            connect_timeout_s=_f(d.get("connect_timeout_s"), base.connect_timeout_s),
            read_timeout_s=_f(d.get("read_timeout_s"), base.read_timeout_s),
            retry_same_max=_i(d.get("retry_same_max"), base.retry_same_max),
            retry_5xx=bool(d.get("retry_5xx", base.retry_5xx)),
            use_retry_after_on_429=bool(d.get("use_retry_after_on_429", base.use_retry_after_on_429)),
        )


@dataclass
class AuthSpec:
    """認証方式。秘密値そのものは持たず、`secret_ref` で vlm_secrets へ問い合わせる。"""
    type: str = "none"          # none | bearer | header_key | query_key
    secret_ref: str = ""        # keyring / env のキー名
    header_name: str = "Authorization"
    query_param: str = "key"

    @classmethod
    def from_mapping(cls, data: dict | None) -> "AuthSpec":
        d = data or {}
        base = cls()
        return cls(
            type=str(d.get("type", base.type) or base.type).lower(),
            secret_ref=str(d.get("secret_ref", "") or ""),
            header_name=str(d.get("header_name", base.header_name) or base.header_name),
            query_param=str(d.get("query_param", base.query_param) or base.query_param),
        )


@dataclass
class VlmConnection:
    connection_id: str
    display_name: str
    kind: ConnectionKind
    protocol: str                       # vlm_protocols.PROTOCOLS のキー
    base_url: str
    model_id: str
    provider_id: str = ""               # 内蔵の場合のみ（gemini / openrouter / cloudflare）
    enabled: bool = True
    auth: AuthSpec = field(default_factory=AuthSpec)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    verify_tls: bool = True
    concurrency: int = 1
    # 内蔵接続が「既知の無料経路」か。カスタムには料金項目を持たせない（常に False 扱い）。
    is_known_free_route: bool = False
    # 有料継続をこの接続で許可したか（内蔵のみ意味を持つ）。
    paid_continuation_allowed: bool = False
    # レスポンス抽出パスの上書き（空ならプロトコル既定）。
    text_path: str = ""
    error_path: str = ""

    @property
    def is_custom(self) -> bool:
        return self.kind in (ConnectionKind.CUSTOM_EXTERNAL, ConnectionKind.CUSTOM_LOCAL)

    @property
    def is_local(self) -> bool:
        return self.kind is ConnectionKind.CUSTOM_LOCAL

    @property
    def free_for_automation(self) -> bool:
        """`free_only` の自動処理へ入れてよいか。

        - 内蔵: 既知の無料経路のみ True
        - カスタムローカル: 料金が発生しないので True（ただし単独選択時のみ使用）
        - カスタム外部: 料金を判定できないため常に False
        """
        if self.kind is ConnectionKind.BUILTIN:
            return self.is_known_free_route
        if self.kind is ConnectionKind.CUSTOM_LOCAL:
            return True
        return False

    @classmethod
    def from_mapping(cls, data: dict) -> "VlmConnection":
        kind_raw = str(data.get("kind", "custom_external")).lower()
        try:
            kind = ConnectionKind(kind_raw)
        except ValueError:
            kind = ConnectionKind.CUSTOM_EXTERNAL
        return cls(
            connection_id=str(data["connection_id"]),
            display_name=str(data.get("display_name", data["connection_id"])),
            kind=kind,
            protocol=str(data.get("protocol", "openai_chat_completions")),
            base_url=str(data.get("base_url", "")),
            model_id=str(data.get("model_id", "")),
            provider_id=str(data.get("provider_id", "")),
            enabled=bool(data.get("enabled", True)),
            auth=AuthSpec.from_mapping(data.get("auth")),
            retry=RetryPolicy.from_mapping(data.get("retry")),
            verify_tls=bool(data.get("verify_tls", True)),
            concurrency=max(1, _i(data.get("concurrency"), 1)),
            is_known_free_route=bool(data.get("is_known_free_route", False)),
            paid_continuation_allowed=bool(data.get("paid_continuation_allowed", False)),
            text_path=str(data.get("response", {}).get("text_path", "") if isinstance(data.get("response"), dict) else data.get("text_path", "")),
            error_path=str(data.get("response", {}).get("error_path", "") if isinstance(data.get("response"), dict) else data.get("error_path", "")),
        )


def _f(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# --- 初期内蔵接続（spec.md 3.1・3.2節） ------------------------------------------------
# base_url と model_id 以外（APIキー等）は起動時にユーザー設定・秘密ストレージから補う。
BUILTIN_CONNECTION_TEMPLATES: list[dict] = [
    {
        "connection_id": "builtin-gemini",
        "display_name": "Gemini API",
        "kind": "builtin",
        "provider_id": "gemini",
        "protocol": "gemini_generate_content",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model_id": "gemma-4-26b-a4b-it",
        "auth": {"type": "header_key", "header_name": "x-goog-api-key", "secret_ref": "vlm/gemini/api_key"},
        "is_known_free_route": True,
    },
    {
        "connection_id": "builtin-openrouter",
        "display_name": "OpenRouter",
        "kind": "builtin",
        "provider_id": "openrouter",
        "protocol": "openai_chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "model_id": "google/gemma-4-26b-a4b-it:free",
        "auth": {"type": "bearer", "secret_ref": "vlm/openrouter/api_key"},
        "is_known_free_route": True,
    },
    {
        "connection_id": "builtin-cloudflare",
        "display_name": "Cloudflare Workers AI",
        "kind": "builtin",
        "provider_id": "cloudflare",
        "protocol": "openai_chat_completions",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "model_id": "@cf/google/gemma-4-26b-a4b-it",
        "auth": {"type": "bearer", "secret_ref": "vlm/cloudflare/api_token"},
        "is_known_free_route": False,
    },
    # --- implement_plan 2.2「後から追加する内蔵候補」。model_id は選択プロファイルの
    # binding から埋める（binding が無ければ build_connection_map が無効化する）。
    {
        "connection_id": "builtin-groq",
        "display_name": "Groq",
        "kind": "builtin",
        "provider_id": "groq",
        "protocol": "openai_chat_completions",
        "base_url": "https://api.groq.com/openai/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/groq/api_key"},
        "is_known_free_route": True,
    },
    {
        "connection_id": "builtin-nvidia",
        "display_name": "NVIDIA NIM",
        "kind": "builtin",
        "provider_id": "nvidia",
        "protocol": "openai_chat_completions",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/nvidia/api_key"},
        "is_known_free_route": True,
    },
    {
        "connection_id": "builtin-mistral",
        "display_name": "Mistral",
        "kind": "builtin",
        "provider_id": "mistral",
        "protocol": "openai_chat_completions",
        "base_url": "https://api.mistral.ai/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/mistral/api_key"},
        "is_known_free_route": True,
    },
    {
        "connection_id": "builtin-huggingface",
        "display_name": "Hugging Face",
        "kind": "builtin",
        "provider_id": "huggingface",
        "protocol": "openai_chat_completions",
        "base_url": "https://router.huggingface.co/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/huggingface/api_token"},
        # Monthly credits exist, but routed inference is metered and can consume paid credits.
        "is_known_free_route": False,
    },
    # OVHcloud は日本居住者によるアカウント作成・実機検証ができなかったため無効化。
    # 対応地域の利用者が接続確認できるまで、内蔵経路として UI へ公開しない。
    # {
    #     "connection_id": "builtin-ovhcloud",
    #     "display_name": "OVHcloud AI Endpoints",
    #     "kind": "builtin",
    #     "provider_id": "ovhcloud",
    #     "protocol": "openai_chat_completions",
    #     "base_url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    #     "model_id": "",
    #     "auth": {"type": "bearer", "secret_ref": "vlm/ovhcloud/api_key"},
    #     "is_known_free_route": False,
    # },
]


def default_builtin_connections() -> list[VlmConnection]:
    return [VlmConnection.from_mapping(t) for t in BUILTIN_CONNECTION_TEMPLATES]
