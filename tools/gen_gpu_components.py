#!/usr/bin/env python3
"""Generate gpu_components.json for a release (Phase 3 of the GPU plan).

Run this ON THE WINDOWS BUILD MACHINE, inside the frozen-build venv that has
`onnxruntime-gpu==<ort_version>` installed (requirements.txt). It:

  1. locates onnxruntime_providers_cuda.dll in that venv's onnxruntime/capi/,
     hashes it and records its size (you host this file yourself, e.g. as a
     GitHub Release asset, and pass its public URL via --providers-url);
  2. for each NVIDIA cu12 runtime package the CUDA EP needs, resolves the latest
     win_amd64 wheel from the PyPI JSON API, downloads it, hashes it, and lists
     the DLLs under */bin/ as members;
  3. writes gpu_components.json to the repo root (picked up by
     pixai_tagger_gui.spec -> _internal/gpu_components.json).

Do NOT commit the generated file - keep it release-local. Verify the wheel
versions match the onnxruntime-gpu CUDA/cuDNN baseline (1.23.x == CUDA 12.x +
cuDNN 9) before publishing.

Usage:
    python tools/gen_gpu_components.py \
        --providers-url https://github.com/OWNER/REPO/releases/download/gpu-runtime-1.23.1/onnxruntime_providers_cuda.dll \
        [--ort-version 1.23.1] [--out gpu_components.json]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

SCHEMA = 1

# cu12 runtime packages the ONNX Runtime CUDA EP loads (see CUDA-ExecutionProvider
# docs / onnxruntime.preload_dlls). Adjust if a future ORT drops/adds one.
NVIDIA_PACKAGES = [
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nvjitlink-cu12",
]


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _find_providers_dll() -> Path:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        sys.exit("onnxruntime-gpu is not installed in this environment.")
    capi = Path(onnxruntime.__file__).resolve().parent / "capi"
    dll = capi / "onnxruntime_providers_cuda.dll"
    if not dll.is_file():
        sys.exit(f"{dll} not found - is this the onnxruntime-gpu (not CPU) package?")
    return dll


def _pypi_latest_win_wheel(package: str) -> tuple[str, str]:
    """(version, wheel_url) for the latest release's win_amd64 wheel."""
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - trusted host
        meta = json.load(resp)
    version = meta["info"]["version"]
    # meta["urls"] is the file list for that latest version (no dependence on the
    # deprecated top-level "releases" map).
    for f in meta.get("urls", []):
        if f.get("packagetype") == "bdist_wheel" and f.get("filename", "").endswith("-win_amd64.whl"):
            return version, f["url"]
    sys.exit(f"{package} {version} has no win_amd64 wheel on PyPI.")


def _wheel_members(whl_bytes: bytes) -> list[dict[str, str]]:
    members: list[dict[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(whl_bytes)) as zf:
        for arc in zf.namelist():
            parts = arc.split("/")
            if len(parts) >= 2 and parts[-2] == "bin" and arc.lower().endswith(".dll"):
                members.append({"arcname": arc, "name": parts[-1]})
    return members


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers-url", required=True,
                    help="public URL where you host onnxruntime_providers_cuda.dll")
    ap.add_argument("--ort-version", default=None,
                    help="override; defaults to the installed onnxruntime version")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "gpu_components.json"))
    args = ap.parse_args()

    dll = _find_providers_dll()
    dll_bytes = dll.read_bytes()
    import onnxruntime
    ort_version = args.ort_version or onnxruntime.__version__

    wheels = []
    for pkg in NVIDIA_PACKAGES:
        pkg_version, wheel_url = _pypi_latest_win_wheel(pkg)
        print(f"  {pkg} {pkg_version}: {wheel_url}")
        with urllib.request.urlopen(wheel_url, timeout=120) as resp:  # noqa: S310
            whl_bytes = resp.read()
        members = _wheel_members(whl_bytes)
        if not members:
            sys.exit(f"{pkg}: no */bin/*.dll members found in wheel.")
        wheels.append({
            "url": wheel_url,
            "sha256": _sha256_bytes(whl_bytes),
            "bytes": len(whl_bytes),
            "members": members,
        })

    spec = {
        "schema": SCHEMA,
        "ort_version": ort_version,
        "direct": [{
            "name": "onnxruntime_providers_cuda.dll",
            "url": args.providers_url,
            "sha256": _sha256_bytes(dll_bytes),
            "bytes": len(dll_bytes),
            "location": "capi",
        }],
        "wheels": wheels,
    }
    Path(args.out).write_text(json.dumps(spec, indent=2), encoding="utf-8")
    total = spec["direct"][0]["bytes"] + sum(w["bytes"] for w in wheels)
    print(f"\nwrote {args.out}  (~{total / 1024 / 1024:.0f} MB total download)")
    print("Remember: host onnxruntime_providers_cuda.dll at --providers-url, "
          "and do NOT commit gpu_components.json.")


if __name__ == "__main__":
    main()
