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

Phase 1（このモジュール）はプラットフォーム非依存のロジックのみ。実際の DL は
Phase 2（gpu_runtime_download）、preload_gpu_dlls の起動時呼び出しも Phase 2。
"""

from __future__ import annotations

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


def gpu_runtime_ready(base_dir: Path | None = None) -> bool:
    """gpu_runtime/ が「使える状態」か検証する。

    gpu_runtime/manifest.json を読み、``required`` に挙がった各ファイル（gpu_runtime/
    からの相対パス）が実在すれば True。JSON パース失敗・キー欠け・型違い・ファイル
    欠けはすべて False（CLAUDE.md #2: is_file() だけで判断しない）。
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
    required = data.get("required")
    if not isinstance(required, list) or not required:
        return False
    for rel in required:
        if not isinstance(rel, str) or not rel:
            return False
        target = root / rel
        try:
            if not target.is_file():
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
    if not gpu_runtime_ready(base_dir):
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
    """gpu_runtime/ が揃っていれば CUDA/cuDNN の DLL を明示ロードする。

    最初の InferenceSession 生成より前に一度だけ呼ぶこと（順序を誤るとシステム
    PATH 上の別バージョン cuDNN を掴む）。Phase 2 で startup から呼び出す。
    例外はすべて握って False（起動を止めない）。
    """
    if not gpu_runtime_ready(base_dir):
        return False
    directory = str(gpu_runtime_dir(base_dir))
    ort_mod = _resolve_ort(ort_module)

    ok = False
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory) and sys.platform.startswith("win"):
        try:
            add_dll_directory(directory)
            ok = True
        except OSError as exc:
            log_dbg(f"preload_gpu_dlls: add_dll_directory failed ({exc!r})")

    preload = getattr(ort_mod, "preload_dlls", None) if ort_mod is not None else None
    if callable(preload):
        try:
            preload(cuda=True, cudnn=True, directory=directory)
            ok = True
        except Exception as exc:  # noqa: BLE001 - ORT raises assorted types here
            log_dbg(f"preload_gpu_dlls: ort.preload_dlls failed ({exc!r})")

    if ok:
        log_dbg(f"preload_gpu_dlls: loaded GPU runtime DLLs from {directory}")
    return ok
