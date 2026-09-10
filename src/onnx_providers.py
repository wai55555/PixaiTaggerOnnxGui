"""ONNX Runtime Execution Provider の解決とフォールバック（GPU 対応の中核）。

設計の背景は docs/260910_gpu_acceleration_options.md、実装計画は
docs/260910_gpu_acceleration_impl_plan.md を参照。

要点:
- 既定は CPU。GPU（CUDA EP）は「GPU コンポーネントが gpu_runtime/ に揃い、
  かつ ONNX Runtime が CUDAExecutionProvider を報告し、かつセッション生成に
  成功した」ときだけ有効化する。
- 上記のどれか一つでも欠ければ CPUExecutionProvider へ静かに落ちる（起動は止めない）。
- onnx_device="cpu"（既定の実効値）のときのセッション生成は、この機能が入る前の
  ``providers=["CPUExecutionProvider"]`` とバイト等価。

このモジュールはプラットフォーム非依存のロジックのみ（実際のダウンロードは
gpu_runtime.GpuRuntimeInstaller、起動時の preload 呼び出しは pixai_tagger_gui.main）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils import log_dbg

if TYPE_CHECKING:
    import onnxruntime as ort  # type: ignore
else:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        ort = None


# gpu_runtime/ は exe 隣（frozen）/ リポジトリルート（source）に置く。model.onnx と同じ層。
GPU_RUNTIME_DIRNAME = "gpu_runtime"
_MANIFEST_NAME = "manifest.json"

CPU_ONLY: list[str] = ["CPUExecutionProvider"]
CUDA_THEN_CPU: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

_VALID_DEVICES = ("auto", "cpu", "cuda")

# ``ort_module`` 引数の「未指定」を表すセンチネル。None は「onnxruntime が無い」を
# 意味する正当な値なので区別する必要がある。
_ORT_DEFAULT = object()


def _resolve_ort(ort_module: Any):
    return ort if ort_module is _ORT_DEFAULT else ort_module


def capi_dir(ort_module: Any = _ORT_DEFAULT) -> Path | None:
    """onnxruntime パッケージの capi/ ディレクトリ（provider DLL の設置先）。

    frozen ビルドでも `onnxruntime/__file__` は `_internal/onnxruntime/__init__.py` を
    指すので、その隣の capi/ が得られる。onnxruntime が無い / 場所を特定できない
    ときは None。
    """
    mod = _resolve_ort(ort_module)
    path = getattr(mod, "__file__", None)
    if not path:
        return None
    try:
        return Path(path).resolve().parent / "capi"
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None


def _base_dir(base_dir: Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    # constants を import すると frozen 判定込みの BASE_DIR が得られる。循環 import を
    # 避けるため関数内 import。
    from constants import BASE_DIR

    return BASE_DIR


def gpu_runtime_dir(base_dir: Path | None = None) -> Path:
    """GPU コンポーネントの設置ディレクトリ（存在は保証しない）。"""
    return _base_dir(base_dir) / GPU_RUNTIME_DIRNAME


def gpu_runtime_ready(base_dir: Path | None = None, *, ort_module: Any = _ORT_DEFAULT) -> bool:
    """gpu_runtime/ が「使える状態」か検証する。

    gpu_runtime/manifest.json を読み、記載された各ファイルが実在すれば True。
    JSON パース失敗・キー欠け・型違い・ファイル欠けはすべて False
    （CLAUDE.md #2: is_file() だけで判断しない）。

    対応する形式:
      - {"files": [{"name": ..., "location": "gpu_runtime"|"capi", "sha256": ...}, ...]}
        gpu_runtime.GpuRuntimeInstaller が書き出す正式形式。location="capi" は
        onnxruntime の capi/ ディレクトリ（provider DLL の設置先）を基準に解決する。
      - {"required": ["a.dll", "b.dll", ...]}（旧形式・すべて gpu_runtime/ 直下）
    起動ごとに走るので SHA-256 は取り直さない（存在確認のみ。ハッシュは
    インストール時に検証済み）。
    """
    root = gpu_runtime_dir(base_dir)
    manifest = root / _MANIFEST_NAME
    try:
        if not manifest.is_file():
            return False
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False

    files = data.get("files")
    if isinstance(files, list) and files:
        capi = capi_dir(ort_module)
        for entry in files:
            if not isinstance(entry, dict):
                return False
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                return False
            location = entry.get("location", "gpu_runtime")
            if location == "capi":
                if capi is None:
                    return False
                target = capi / name
            else:
                target = root / name
            try:
                if not target.is_file():
                    return False
            except OSError:
                return False
        return True

    required = data.get("required")
    if not isinstance(required, list) or not required:
        return False
    for rel in required:
        if not isinstance(rel, str) or not rel:
            return False
        try:
            if not (root / rel).is_file():
                return False
        except OSError:
            return False
    return True


def _normalize_device(prefer: Any) -> str:
    value = str(prefer).strip().lower() if prefer is not None else ""
    return value if value in _VALID_DEVICES else "auto"


def resolve_providers(
    prefer: str,
    *,
    ort_module: Any = _ORT_DEFAULT,
    base_dir: Path | None = None,
) -> list[str]:
    """config.ini [Behavior] onnx_device の値から InferenceSession の providers を決める。

    prefer: "auto" | "cpu" | "cuda"（空文字・不正値・None は "auto" 扱い）。
      - "cpu": 常に CPU。
      - "auto": GPU が使えそうなら CUDA、駄目なら CPU（黙って）。
      - "cuda": GPU を積極的に使う。条件未達なら CPU に落ち、warning を残す
               （利用者が明示的に選んだのに効いていない、と分かるように）。
    """
    device = _normalize_device(prefer)
    if device == "cpu":
        return list(CPU_ONLY)

    ort_mod = _resolve_ort(ort_module)
    if ort_mod is None:
        if device == "cuda":
            log_dbg("resolve_providers: onnx_device=cuda but onnxruntime is unavailable; using CPU")
        return list(CPU_ONLY)

    try:
        available = set(ort_mod.get_available_providers())
    except Exception:  # pragma: no cover - defensive
        available = set()

    reasons: list[str] = []
    if "CUDAExecutionProvider" not in available:
        reasons.append("CUDAExecutionProvider not offered by this onnxruntime build")
    # gpu_runtime_ready() は「凍結 exe が自前 DL した CUDA/cuDNN が揃っているか」。
    # onnx_device="cuda" は利用者が自分で CUDA を用意した宣言（例: pip install
    # onnxruntime-gpu + nvidia-*）なので、このゲートは課さず素直に CUDA を試す
    # （駄目なら make_session が CPU へフォールバックし warning を残す）。
    # "auto" は既定なので保守的に、DL 済みのときだけ opt-in する。
    if device != "cuda" and not gpu_runtime_ready(base_dir, ort_module=ort_mod):
        reasons.append("gpu_runtime/ not present or incomplete")

    if not reasons:
        return list(CUDA_THEN_CPU)

    if device == "cuda":
        log_dbg("resolve_providers: onnx_device=cuda requested but falling back to CPU (" + "; ".join(reasons) + ")")
    return list(CPU_ONLY)


def make_session(
    model_path: Path | str,
    *,
    sess_options: Any = None,
    prefer: str = "auto",
    ort_module: Any = _ORT_DEFAULT,
    base_dir: Path | None = None,
    label: str = "session",
):
    """InferenceSession を生成する。GPU を要求して失敗したら CPU で作り直す。

    ``prefer`` が CPU 専用に解決された場合の呼び出しは、この機能が入る前の
    ``ort.InferenceSession(str(model_path), sess_options=sess_options,
    providers=["CPUExecutionProvider"])`` とバイト等価。
    """
    ort_mod = _resolve_ort(ort_module)
    if ort_mod is None:
        raise ImportError("onnxruntime is not available")

    providers = resolve_providers(prefer, ort_module=ort_mod, base_dir=base_dir)
    model_str = str(model_path)
    try:
        session = ort_mod.InferenceSession(model_str, sess_options=sess_options, providers=providers)
    except Exception as exc:
        if providers == list(CPU_ONLY):
            raise
        log_dbg(f"{label}: session creation with {providers} failed ({exc!r}); retrying on CPU")
        session = ort_mod.InferenceSession(model_str, sess_options=sess_options, providers=list(CPU_ONLY))

    try:
        active = list(session.get_providers())
    except Exception:  # pragma: no cover - defensive
        active = []
    if active:
        log_dbg(f"{label}: active execution provider = {active[0]}")
    return session


def preload_gpu_dlls(base_dir: Path | None = None, ort_module: Any = _ORT_DEFAULT) -> bool:
    """CUDA/cuDNN の DLL を最初の InferenceSession より前にロードする。

    最初の InferenceSession 生成より前に一度だけ呼ぶこと（順序を誤るとシステム
    PATH 上の別バージョン cuDNN を掴む）。`main()` の冒頭で呼ぶ。

    - `gpu_runtime/` が揃っていれば（＝凍結 exe が自前 DL 済み）そこからロード。
    - 揃っていなければ、pip で入れた `nvidia-*` wheel（ソース実行の開発者向け）を
      `ort.preload_dlls()` 既定探索で拾う。凍結 exe で未 DL の場合はどちらも
      no-op（`resolve_providers("auto")` が CPU を返すので実害なし）。
    例外はすべて握って返り値で表現（起動は止めない）。
    """
    ort_mod = _resolve_ort(ort_module)
    preload = getattr(ort_mod, "preload_dlls", None) if ort_mod is not None else None
    if not callable(preload):
        return False

    if gpu_runtime_ready(base_dir, ort_module=ort_mod):
        directory = str(gpu_runtime_dir(base_dir))
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory) and sys.platform.startswith("win"):
            try:
                add_dll_directory(directory)
            except OSError as exc:
                log_dbg(f"preload_gpu_dlls: add_dll_directory failed ({exc!r})")
        try:
            preload(cuda=True, cudnn=True, directory=directory)
            log_dbg(f"preload_gpu_dlls: loaded GPU runtime DLLs from {directory}")
            return True
        except Exception as exc:  # noqa: BLE001 - ORT raises assorted types here
            log_dbg(f"preload_gpu_dlls: ort.preload_dlls(directory=...) failed ({exc!r})")
            return False

    # Only fall through to the default search when the pip nvidia-* wheels are
    # actually present (they create the `nvidia` namespace package). Otherwise
    # ort.preload_dlls() prints a wall of "Failed to load cudnn64_9.dll ... Please
    # follow ... install CUDA" to stderr on every launch of a plain
    # `pip install onnxruntime-gpu` env, which is just CPU-mode noise.
    nvidia_bin_dirs = _pip_nvidia_bin_dirs()
    if not nvidia_bin_dirs:
        return False
    # Put every nvidia-*/bin on the DLL search path. ort.preload_dlls() only
    # preloads a fixed list of cuDNN DLLs by absolute path; cuDNN 9 then does its
    # own LoadLibrary of sublibraries it doesn't know about (cudnn_engines_tensor_ir
    # 64_9.dll, cudnn_ext64_9.dll, added in 9.13+), which fails with
    # CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED unless their directory is searchable.
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory):
        for d in nvidia_bin_dirs:
            try:
                add_dll_directory(d)
            except OSError:
                pass
    try:
        preload(cuda=True, cudnn=True)
        log_dbg("preload_gpu_dlls: preloaded CUDA/cuDNN from the pip nvidia-* wheels")
        return True
    except Exception as exc:  # noqa: BLE001
        log_dbg(f"preload_gpu_dlls: default ort.preload_dlls() failed ({exc!r})")
        return False


def _pip_nvidia_bin_dirs() -> list[str]:
    """Directories of the shared libs shipped by the pip `nvidia-*-cu12` wheels, if
    installed: `nvidia/<lib>/bin` on Windows, `nvidia/<lib>/lib` on Linux.

    Empty list when no such wheel is present (plain `pip install onnxruntime-gpu`).
    """
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    roots = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    out: list[str] = []
    for root in roots:
        for pat in ("*/bin", "*/lib"):
            try:
                out.extend(str(sub) for sub in Path(root).glob(pat) if sub.is_dir())
            except OSError:
                continue
    return out
