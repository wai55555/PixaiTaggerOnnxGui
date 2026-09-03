"""VLM モデルプロファイルと同一性判定（260901_VLM_spec.md 3章 / design.md 4.1・5.1節）。

内蔵サービス間のフォールバックは「サービス」ではなく「同一モデルプロファイル」を単位に
する。サービスごとの呼び名の違いは対応表（bindings）で吸収し、`vlm_protocols.py` などの
通信コードには分散させない。量子化方式・ベースモデル・リビジョンが違うものは同一モデルと
みなさない（別モデルへの自動切り替えは一切行わない）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

_TOKEN_SPLIT = re.compile(r"[/:@_.\-\s]+")
# 「同一モデル」を判定するのに効くトークン（サイズ・世代・ファミリー）。汎用語は除外。
_STOP_TOKENS = frozenset({"", "free", "instruct", "it", "chat", "latest", "preview",
                          "vision", "vl", "model", "google", "meta", "cf", "api"})


def _sig_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_SPLIT.split((text or "").lower()) if t and t not in _STOP_TOKENS}


class ModelIdentityStatus(str, Enum):
    """内蔵サービスが提供するモデルが、プロファイルの正規モデルと同一と言えるか。

    - VERIFIED: 厳格フォールバックで使用可能
    - DECLARED: 同一と宣言されているが、厳格フォールバックでは使わない（警告対象）
    - UNKNOWN : 判定不能。厳格フォールバックから除外
    """
    VERIFIED = "verified"
    DECLARED = "declared"
    UNKNOWN = "unknown"


def parse_identity_status(raw: object) -> ModelIdentityStatus:
    """手書き JSON からの値を検証する。不明・型違いは UNKNOWN（安全側）へ。"""
    try:
        return ModelIdentityStatus(str(raw).strip().lower())
    except (ValueError, AttributeError):
        return ModelIdentityStatus.UNKNOWN


@dataclass(frozen=True)
class ProviderConstraint:
    """OpenRouter のように内部プロバイダーが変わりうるサービスの固定条件。

    承認済みプロバイダーへ固定でき、かつ自動プロバイダーフォールバックを無効化できた
    場合だけ VERIFIED を名乗れる（spec.md 3.2節）。
    """
    allowed_providers: tuple[str, ...] = ()
    allow_fallbacks: bool = True

    @property
    def is_pinned(self) -> bool:
        return bool(self.allowed_providers) and not self.allow_fallbacks


@dataclass(frozen=True)
class ModelBinding:
    """1つの内蔵プロバイダーにおける、このモデルプロファイルの実体。"""
    provider_id: str
    model_id: str
    identity_status: ModelIdentityStatus = ModelIdentityStatus.UNKNOWN
    provider_constraint: ProviderConstraint | None = None
    # この経路が「既知の無料経路」か（プロバイダーではなく binding 単位。OpenRouter の
    # `:free` サフィックスや、無料枠のあるサービスなど）。connection map 構築時に反映。
    free_route: bool = False

    def effective_identity_status(self) -> ModelIdentityStatus:
        """provider_constraint を固定できていない場合、VERIFIED を格下げする。"""
        status = self.identity_status
        if status is ModelIdentityStatus.VERIFIED and self.provider_constraint is not None:
            if not self.provider_constraint.is_pinned:
                return ModelIdentityStatus.DECLARED
        return status

    def is_strict_fallback_eligible(self) -> bool:
        """通常（厳格）フォールバックの候補になれるか。VERIFIED のみ True。"""
        return self.effective_identity_status() is ModelIdentityStatus.VERIFIED


@dataclass(frozen=True)
class VlmModelProfile:
    """利用者が選ぶのはサービス名ではなくこのプロファイル（requirement FR-003）。"""
    profile_id: str
    display_name: str
    canonical_model_id: str
    family: str = ""
    base_model: str = ""
    revision: str = ""
    # unquantized_required / int8 / awq / gguf-q4 など。空・不明は厳格判定から除外する材料。
    quantization: str = "unknown"
    bindings: dict[str, ModelBinding] = field(default_factory=dict)
    # 別名（サービスの生モデルID等）から profile_id を引くための逆引き補助。
    aliases: tuple[str, ...] = ()

    def binding_for(self, provider_id: str) -> ModelBinding | None:
        return self.bindings.get(provider_id)

    def strict_fallback_providers(self) -> list[str]:
        """厳格フォールバックの対象になる provider_id を、bindings 挿入順で返す。"""
        return [pid for pid, b in self.bindings.items() if b.is_strict_fallback_eligible()]

    def quantization_is_strict(self) -> bool:
        """量子化方式が厳格判定に足るか（不明・空は不可。spec.md 3章 / plan 3.1節）。"""
        q = self.quantization.strip().lower()
        return bool(q) and q != "unknown"


class VlmModelRegistry:
    """内蔵モデルプロファイルの集合。model_registry.py（ローカル ONNX 用）とは混在させない。"""

    def __init__(self, profiles: list[VlmModelProfile] | None = None):
        self._profiles: dict[str, VlmModelProfile] = {}
        for p in profiles or []:
            self.add(p)

    def add(self, profile: VlmModelProfile) -> None:
        self._profiles[profile.profile_id] = profile

    def get(self, profile_id: str) -> VlmModelProfile | None:
        return self._profiles.get(profile_id)

    def all_profiles(self) -> list[VlmModelProfile]:
        return list(self._profiles.values())

    def resolve_alias(self, name: str) -> VlmModelProfile | None:
        """profile_id / canonical_model_id / alias / いずれかの binding.model_id で引く。"""
        key = name.strip()
        if key in self._profiles:
            return self._profiles[key]
        low = key.lower()
        for p in self._profiles.values():
            if p.canonical_model_id.lower() == low:
                return p
            if any(a.lower() == low for a in p.aliases):
                return p
            if any(b.model_id.lower() == low for b in p.bindings.values()):
                return p
        return None


# --- 内蔵プロファイル（spec.md 3章 / implement_plan 2.1・2.2・18章） ------------------
# identity_status は控えめに置く（多くは DECLARED、実在が未確認のものは UNKNOWN）。
# 各プロバイダーでの正確なモデル ID は「接続診断／1枚テスト」を通すまで確定しない前提。
# 診断のフル PASS、認証済み429／診断上限による到達確認、または生成成功で
# `[Vlm] verified_bindings` に載り VERIFIED 扱いになる（vlm_config._apply_verified_promotions）。
#
# providers: gemini / openrouter / cloudflare / groq / nvidia / mistral /
#            huggingface / vercel / openai / anthropic
#            （ovhcloud は実機検証できるまで無効）
#   - gemini      : Google Generative Language API（gemini_generate_content）
#   - openrouter  : OpenRouter（openai_chat_completions、`:free` サフィックスで無料経路）
#   - cloudflare  : Cloudflare Workers AI（openai_chat_completions、要 account_id）
#   - groq        : Groq（openai_chat_completions、無料枠あり）
#   - nvidia      : NVIDIA NIM / build.nvidia.com（openai_chat_completions、無料クレジット）
#   - mistral     : Mistral La Plateforme（openai_chat_completions、無料枠あり）
#   - huggingface : Hugging Face Inference Providers（openai_chat_completions、従量課金）
#   - vercel      : Vercel AI Gateway（openai_chat_completions、従量課金）
#   - openai      : OpenAI Responses API（従量課金）
#   - anthropic   : Anthropic Messages API（従量課金）
#   - ovhcloud    : OVHcloud AI Endpoints（日本居住者環境で実機検証できるまで無効）

GEMMA_4_26B_A4B_IT = VlmModelProfile(
    profile_id="gemma-4-26b-a4b-it",
    display_name="Gemma 4 26B A4B IT",
    canonical_model_id="gemma-4-26b-a4b-it",
    family="Gemma 4",
    base_model="gemma-4-26b-a4b-it",
    revision="provider_verified",
    quantization="unquantized_required",
    aliases=(
        "google/gemma-4-26b-a4b-it",
        "google/gemma-4-26b-a4b-it:free",
        "@cf/google/gemma-4-26b-a4b-it",
    ),
    bindings={
        "gemini": ModelBinding("gemini", "gemma-4-26b-a4b-it",
                               ModelIdentityStatus.DECLARED, free_route=True),
        "openrouter": ModelBinding("openrouter", "google/gemma-4-26b-a4b-it:free",
                                   ModelIdentityStatus.DECLARED,
                                   ProviderConstraint(allowed_providers=(), allow_fallbacks=True),
                                   free_route=True),
        "cloudflare": ModelBinding("cloudflare", "@cf/google/gemma-4-26b-a4b-it",
                                   ModelIdentityStatus.DECLARED),
        "huggingface": ModelBinding("huggingface", "google/gemma-4-26B-A4B-it",
                                     ModelIdentityStatus.UNKNOWN),
        "vercel": ModelBinding("vercel", "google/gemma-4-26b-a4b-it",
                                ModelIdentityStatus.DECLARED),
    },
)

# --- 2.2 「後から追加する内蔵候補」。実モデル ID は接続で確定させる前提なので UNKNOWN。
GEMMA_4_31B_IT = VlmModelProfile(
    profile_id="gemma-4-31b-it",
    display_name="Gemma 4 31B IT",
    canonical_model_id="gemma-4-31b-it",
    family="Gemma 4",
    base_model="gemma-4-31b-it",
    quantization="unknown",
    aliases=("google/gemma-4-31b-it", "google/gemma-4-31b-it:free", "@cf/google/gemma-4-31b-it"),
    bindings={
        "gemini": ModelBinding("gemini", "gemma-4-31b-it", ModelIdentityStatus.UNKNOWN),
        "openrouter": ModelBinding("openrouter", "google/gemma-4-31b-it:free",
                                   ModelIdentityStatus.UNKNOWN, free_route=True),
        "cloudflare": ModelBinding("cloudflare", "@cf/google/gemma-4-31b-it",
                                   ModelIdentityStatus.UNKNOWN),
        "nvidia": ModelBinding("nvidia", "google/gemma-4-31b-it", ModelIdentityStatus.UNKNOWN,
                               free_route=True),
        "groq": ModelBinding("groq", "gemma-4-31b-it", ModelIdentityStatus.UNKNOWN, free_route=True),
        "huggingface": ModelBinding("huggingface", "google/gemma-4-31B-it",
                                     ModelIdentityStatus.UNKNOWN),
        "vercel": ModelBinding("vercel", "google/gemma-4-31b-it",
                                ModelIdentityStatus.DECLARED),
    },
)

QWEN3_8_27B = VlmModelProfile(
    profile_id="qwen3.8-27b",
    display_name="Qwen3.8 27B",
    canonical_model_id="qwen3.8-27b",
    family="Qwen3.8",
    base_model="qwen3.8-27b",
    quantization="unknown",
    aliases=("qwen/qwen3.8-27b", "qwen/qwen3.8-27b-instruct", "qwen3.8-27b-instruct"),
    bindings={
        "openrouter": ModelBinding("openrouter", "qwen/qwen3.8-27b", ModelIdentityStatus.UNKNOWN,
                                   free_route=True),
        "nvidia": ModelBinding("nvidia", "qwen/qwen3.8-27b-instruct", ModelIdentityStatus.UNKNOWN,
                               free_route=True),
        "groq": ModelBinding("groq", "qwen3.8-27b", ModelIdentityStatus.UNKNOWN, free_route=True),
        # "ovhcloud": ModelBinding("ovhcloud", "Qwen3.8-27B",
        #                            ModelIdentityStatus.UNKNOWN, free_route=True),
    },
)

QWEN3_6_27B = VlmModelProfile(
    profile_id="qwen3.6-27b",
    display_name="Qwen3.6 27B",
    canonical_model_id="qwen3.6-27b",
    family="Qwen3.6",
    base_model="qwen3.6-27b",
    quantization="unknown",
    aliases=("qwen/qwen3.6-27b", "qwen/qwen3.6-27b-instruct", "qwen3.6-27b-instruct"),
    bindings={
        "openrouter": ModelBinding("openrouter", "qwen/qwen3.6-27b", ModelIdentityStatus.UNKNOWN,
                                   free_route=True),
        "nvidia": ModelBinding("nvidia", "qwen/qwen3.6-27b-instruct", ModelIdentityStatus.UNKNOWN,
                               free_route=True),
        "groq": ModelBinding("groq", "qwen3.6-27b", ModelIdentityStatus.UNKNOWN, free_route=True),
        # "ovhcloud": ModelBinding("ovhcloud", "Qwen3.6-27B",
        #                            ModelIdentityStatus.UNKNOWN),
    },
)

# Mistral の VLM（Pixtral 系）。実 ID は接続で確認する。
PIXTRAL_12B = VlmModelProfile(
    profile_id="pixtral-12b",
    display_name="Pixtral 12B",
    canonical_model_id="pixtral-12b",
    family="Pixtral",
    base_model="pixtral-12b",
    quantization="unknown",
    aliases=("mistralai/pixtral-12b", "pixtral-12b-2409"),
    bindings={
        "mistral": ModelBinding("mistral", "pixtral-12b-2409", ModelIdentityStatus.UNKNOWN,
                                free_route=True),
        "openrouter": ModelBinding("openrouter", "mistralai/pixtral-12b", ModelIdentityStatus.UNKNOWN,
                                   free_route=True),
    },
)

OPENAI_GPT_5_6_LUNA = VlmModelProfile(
    profile_id="openai-gpt-5.6-luna",
    display_name="OpenAI GPT-5.6 Luna",
    canonical_model_id="gpt-5.6-luna",
    family="GPT-5.6",
    base_model="gpt-5.6-luna",
    revision="provider_managed",
    quantization="provider_managed",
    aliases=("openai/gpt-5.6-luna",),
    bindings={
        "openai": ModelBinding("openai", "gpt-5.6-luna", ModelIdentityStatus.DECLARED),
        "vercel": ModelBinding("vercel", "openai/gpt-5.6-luna", ModelIdentityStatus.DECLARED),
    },
)

CLAUDE_HAIKU_4_5 = VlmModelProfile(
    profile_id="claude-haiku-4-5",
    display_name="Claude Haiku 4.5",
    canonical_model_id="claude-haiku-4-5-20251001",
    family="Claude 4.5",
    base_model="claude-haiku-4-5-20251001",
    revision="20251001",
    quantization="provider_managed",
    aliases=("anthropic/claude-haiku-4.5",),
    bindings={
        "anthropic": ModelBinding("anthropic", "claude-haiku-4-5-20251001",
                                  ModelIdentityStatus.DECLARED),
        "vercel": ModelBinding("vercel", "anthropic/claude-haiku-4.5",
                                ModelIdentityStatus.DECLARED),
    },
)

_ALL_PROFILES = [
    GEMMA_4_26B_A4B_IT, GEMMA_4_31B_IT, QWEN3_8_27B, QWEN3_6_27B, PIXTRAL_12B,
    OPENAI_GPT_5_6_LUNA, CLAUDE_HAIKU_4_5,
]


def _profile_reference_tokens(profile: VlmModelProfile) -> set[str]:
    """このプロファイルの「同一モデル」を表す代表トークン集合。"""
    toks: set[str] = set()
    for s in (profile.canonical_model_id, profile.base_model, profile.family):
        toks |= _sig_tokens(s)
    for a in profile.aliases:
        toks |= _sig_tokens(a)
    return toks


def match_model_id(profile: VlmModelProfile, provider_id: str,
                   candidates: list[str]) -> tuple[str | None, float]:
    """プロバイダーが返したモデル ID 一覧から、このプロファイルの正規モデルに
    もっとも合致するものを選ぶ。戻り値は (best_id | None, score[0..1])。

    フォールバックは「同一モデルを複数プロバイダーで回す」設計なので、ここでの狙いは
    「一覧の中でプロファイルのモデルはどれか」を当てること。任意モデルの自由選択ではない。
    """
    if not candidates:
        return None, 0.0
    binding = profile.binding_for(provider_id)
    lowered = {c.lower(): c for c in candidates}
    # 1) binding の既定 ID / alias が一覧にそのままあれば即決。
    exacts = [binding.model_id] if binding is not None else []
    exacts += list(profile.aliases) + [profile.canonical_model_id]
    for e in exacts:
        if e and e.lower() in lowered:
            return lowered[e.lower()], 1.0
    # 2) トークン一致で採点。
    ref = _profile_reference_tokens(profile)
    if not ref:
        return None, 0.0
    best_id, best_score = None, 0.0
    for c in candidates:
        ct = _sig_tokens(c)
        if not ct:
            continue
        inter = ref & ct
        # 数字トークン（27b / 12b など）が食い違うなら別サイズ＝別モデル。強く減点。
        ref_nums = {t for t in ref if any(ch.isdigit() for ch in t)}
        cand_nums = {t for t in ct if any(ch.isdigit() for ch in t)}
        size_ok = not (ref_nums and cand_nums) or bool(ref_nums & cand_nums)
        score = len(inter) / len(ref | ct)
        if not size_ok:
            score *= 0.3
        if score > best_score:
            best_id, best_score = c, score
    return (best_id, best_score) if best_score >= 0.34 else (None, best_score)


def looks_same_family(profile: VlmModelProfile, model_id: str) -> bool:
    """model_id がプロファイルと「だいたい同じモデル」に見えるか（緩いチェック）。"""
    if not model_id:
        return False
    ref = _profile_reference_tokens(profile)
    ct = _sig_tokens(model_id)
    if not ref or not ct:
        return False
    ref_nums = {t for t in ref if any(ch.isdigit() for ch in t)}
    cand_nums = {t for t in ct if any(ch.isdigit() for ch in t)}
    if ref_nums and cand_nums and not (ref_nums & cand_nums):
        return False
    return len(ref & ct) >= 1


# 現在のVLM UIで取得できても、画像入力を受け付けないことが公式仕様で明確な
# モデル群。GroqのCompoundはテキスト／ツール用で、画像キャプション経路には使えない。
# 将来のモデル追加で一覧から漏れても、既知の非VLMだけは安全側で弾く。
_KNOWN_NON_VISION_MODEL_IDS = {
    "groq": frozenset({
        "groq/compound",
        "groq/compound-mini",
        "compound",
        "compound-mini",
        "compound-beta",
        "compound-beta-mini",
        "groq/compound-beta",
        "groq/compound-beta-mini",
    }),
}

# 選択プロファイルに binding がない場合でも、モデル一覧からVLMを探せるようにするための
# 既知の画像対応ID。通常のプロファイルでは binding のIDを優先して判定する。
# Groqの現行Visionドキュメントに掲載されている画像対応モデルもここへ明示する。
_KNOWN_VISION_MODEL_IDS = {
    "groq": frozenset({
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b",
    }),
}


def is_known_non_vision_model(provider_id: str, model_id: str) -> bool:
    """既知のテキスト専用モデルかを判定する。"""
    provider = (provider_id or "").strip().lower()
    model = (model_id or "").strip().lower()
    return model in _KNOWN_NON_VISION_MODEL_IDS.get(provider, ())


def _known_vision_model_ids(provider_id: str) -> set[str]:
    """内蔵プロファイルとプロバイダー公式掲載IDから既知のVLM IDを集める。"""
    provider = (provider_id or "").strip().lower()
    ids = set(_KNOWN_VISION_MODEL_IDS.get(provider, ()))
    for profile in _ALL_PROFILES:
        binding = profile.binding_for(provider)
        if binding is not None and binding.model_id:
            ids.add(binding.model_id.strip().lower())
    return ids


def is_vlm_model_id(profile: VlmModelProfile | None, provider_id: str,
                    model_id: str) -> bool:
    """選択中のVLMプロファイルで利用可能なモデルIDかを判定する。

    選択プロファイルに binding がある経路は、同じモデルに対応するIDだけを許可する。
    binding がない経路は、既知のVLM IDから選べるようにする。どちらの場合も、既知の
    テキスト専用モデルは許可しない。
    """
    if not (model_id or "").strip() or is_known_non_vision_model(provider_id, model_id):
        return False
    if profile is not None and profile.binding_for(provider_id) is not None:
        return looks_same_family(profile, model_id)
    return (model_id or "").strip().lower() in _known_vision_model_ids(provider_id)


def filter_vlm_model_ids(profile: VlmModelProfile | None, provider_id: str,
                         candidates: list[str]) -> list[str]:
    """モデル一覧から、選択中プロファイルで使えるVLM IDだけを返す。"""
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        model_id = str(candidate or "").strip()
        if model_id and model_id not in seen and is_vlm_model_id(profile, provider_id, model_id):
            seen.add(model_id)
            out.append(model_id)
    return out


def default_registry() -> VlmModelRegistry:
    return VlmModelRegistry(_ALL_PROFILES)


DEFAULT_MODEL_PROFILE_ID = GEMMA_4_26B_A4B_IT.profile_id
