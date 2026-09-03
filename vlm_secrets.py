"""API キーなど秘密情報の解決（260901_VLM_spec.md 15章 / implement_plan 13.1節）。

優先順位: OS の秘密情報ストレージ（keyring） → 環境変数 → セッション中だけ保持。
秘密値は config.ini / vlm_connections.json / 設定エクスポート / debug log / エラー
ダイアログのいずれにも出さない。この層だけが実体を扱う。
"""
from __future__ import annotations

import os
import threading

from utils import write_debug_log

try:  # keyring はオプション依存。無い環境ではセッション保持のみに落ちる。
    import keyring  # type: ignore
except Exception:  # pragma: no cover - 環境依存
    keyring = None  # type: ignore

_SERVICE = "PixaiTaggerOnnxGui.VLM"


def _load_dotenv() -> None:
    """exe/スクリプトの隣、または作業ディレクトリの `.env` を環境変数へ読み込む。

    内蔵プロバイダーぶんのキーをダイアログで登録する代わりに `.env` 一枚で済ませられる。
    既に設定済みの環境変数は上書きしない。値の `"..."` / `'...'` は外す。
    """
    try:
        from constants import BASE_DIR
        candidates = [BASE_DIR / ".env", os.path.join(os.getcwd(), ".env")]
    except Exception:  # pragma: no cover
        candidates = [os.path.join(os.getcwd(), ".env")]
    seen: set[str] = set()
    for path in candidates:
        p = str(path)
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:  # pragma: no cover
            write_debug_log(f"vlm_secrets: cannot read {p}: {e}")
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:]
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

# secret_ref -> 環境変数名 の対応（既知の内蔵接続用）。未知の ref は
# "PIXAI_VLM_" + ref を英大文字化した名前も試す。
_ENV_ALIASES = {
    "vlm/gemini/api_key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "vlm/openrouter/api_key": ("OPENROUTER_API_KEY",),
    "vlm/cloudflare/api_token": ("CLOUDFLARE_API_TOKEN", "CF_API_TOKEN"),
    "vlm/groq/api_key": ("GROQ_API_KEY",),
    "vlm/nvidia/api_key": ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
    "vlm/mistral/api_key": ("MISTRAL_API_KEY",),
    "vlm/huggingface/api_token": ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    # "vlm/ovhcloud/api_key": ("OVH_AI_ENDPOINTS_ACCESS_TOKEN", "OVHCLOUD_API_KEY"),
}

_lock = threading.Lock()
_session_store: dict[str, str] = {}


def _env_candidates(secret_ref: str) -> list[str]:
    names = list(_ENV_ALIASES.get(secret_ref, ()))
    generic = "PIXAI_" + secret_ref.replace("/", "_").replace("-", "_").upper()
    if generic not in names:
        names.append(generic)
    return names


def get_secret(secret_ref: str) -> str | None:
    """secret_ref に対応する秘密値を返す。見つからなければ None。"""
    if not secret_ref:
        return None
    # 1. keyring
    if keyring is not None:
        try:
            v = keyring.get_password(_SERVICE, secret_ref)
            if v:
                return v
        except Exception as e:  # backend 無し等
            write_debug_log(f"vlm_secrets: keyring get failed for a ref: {type(e).__name__}")
    # 2. 環境変数
    for name in _env_candidates(secret_ref):
        v = os.environ.get(name)
        if v:
            return v
    # 3. セッション
    with _lock:
        return _session_store.get(secret_ref)


def set_secret(secret_ref: str, value: str, *, persist: bool) -> bool:
    """秘密値を保存する。persist=True なら keyring、False ならセッションのみ。

    戻り値は「keyring へ永続保存できたか」。persist=False のとき、および keyring 保存に
    失敗してセッション保持へフォールバックしたときは False。
    """
    if not secret_ref:
        return False
    if persist and keyring is not None:
        try:
            keyring.set_password(_SERVICE, secret_ref, value)
            with _lock:
                _session_store.pop(secret_ref, None)
            return True
        except Exception as e:
            write_debug_log(f"vlm_secrets: keyring set failed, falling back to session: {type(e).__name__}")
    with _lock:
        _session_store[secret_ref] = value
    return False


def delete_secret(secret_ref: str) -> None:
    if not secret_ref:
        return
    if keyring is not None:
        try:
            keyring.delete_password(_SERVICE, secret_ref)
        except Exception:
            pass
    with _lock:
        _session_store.pop(secret_ref, None)


def secret_status(secret_ref: str) -> str:
    """UI 表示用: 秘密値の在り処。'keyring' / 'env' / 'session' / 'missing'。"""
    if not secret_ref:
        return "missing"
    if keyring is not None:
        try:
            if keyring.get_password(_SERVICE, secret_ref):
                return "keyring"
        except Exception:
            pass
    for name in _env_candidates(secret_ref):
        if os.environ.get(name):
            return "env"
    with _lock:
        if secret_ref in _session_store:
            return "session"
    return "missing"


_keyring_probe: bool | None = None


def keyring_available() -> bool:
    """OS の秘密情報ストレージが「実際に使えるか」。

    `import keyring` が通っても、バックエンドが無い環境では
    `keyring.backends.fail.Keyring`（priority<=0）が返り、`set_password` は例外になる。
    モジュールの有無ではなく、使えるバックエンドがあるかを見る（結果は一度だけ判定）。
    """
    global _keyring_probe
    if _keyring_probe is not None:
        return _keyring_probe
    if keyring is None:
        _keyring_probe = False
        return False
    try:
        backend = keyring.get_keyring()
        module = (type(backend).__module__ or "").lower()
        priority = float(getattr(backend, "priority", 1) or 0)
        _keyring_probe = ("fail" not in module) and priority > 0
    except Exception as e:  # pragma: no cover - 環境依存
        write_debug_log(f"vlm_secrets: keyring backend probe failed: {type(e).__name__}")
        _keyring_probe = False
    return _keyring_probe
