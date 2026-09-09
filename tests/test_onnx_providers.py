"""onnx_providers の EP 解決 / フォールバックと、[Behavior] onnx_device・gpu_setup_prompt
の config 往復テスト（Phase 1: docs/260910_gpu_acceleration_impl_plan.md）。

Offline only - no network, no GUI, no real onnxruntime.
Run:  rtk pytest tests/test_onnx_providers.py -q
"""
import configparser
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import onnx_providers as OP
import app_settings as A


# --- fakes -----------------------------------------------------------------

class _FakeSession:
    def __init__(self, model_path, sess_options=None, providers=None):
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = list(providers or [])

    def get_providers(self):
        return list(self.providers)


class _FakeOrt:
    """get_available_providers と InferenceSession だけを持つ最小 onnxruntime 代役。"""

    def __init__(self, available=("CPUExecutionProvider",), fail_on=()):
        self._available = list(available)
        self._fail_on = set(fail_on)  # {"cuda", "cpu"}
        self.calls: list[list[str]] = []

    def get_available_providers(self):
        return list(self._available)

    def InferenceSession(self, model_path, sess_options=None, providers=None):
        provs = list(providers or [])
        self.calls.append(provs)
        if provs[:1] == ["CUDAExecutionProvider"] and "cuda" in self._fail_on:
            raise RuntimeError("fake CUDA init failure")
        if provs == ["CPUExecutionProvider"] and "cpu" in self._fail_on:
            raise RuntimeError("fake CPU init failure")
        return _FakeSession(model_path, sess_options, provs)


def _make_gpu_runtime(base: Path, complete: bool = True) -> Path:
    d = base / OP.GPU_RUNTIME_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    (d / "onnxruntime_providers_cuda.dll").write_bytes(b"x")
    if complete:
        (d / "cudnn64_9.dll").write_bytes(b"x")
    (d / "manifest.json").write_text(
        json.dumps({"required": ["onnxruntime_providers_cuda.dll", "cudnn64_9.dll"]}),
        encoding="utf-8",
    )
    return d


@pytest.fixture
def caplog_dbg(monkeypatch):
    msgs: list[str] = []
    monkeypatch.setattr(OP, "log_dbg", lambda m, *a, **k: msgs.append(m))
    return msgs


# --- resolve_providers ---------------------------------------------------------

def test_prefer_cpu_never_touches_cuda(tmp_path):
    _make_gpu_runtime(tmp_path)
    ort = _FakeOrt(available=("CPUExecutionProvider", "CUDAExecutionProvider"))
    assert OP.resolve_providers("cpu", ort_module=ort, base_dir=tmp_path) == OP.CPU_ONLY


def test_ort_missing_is_cpu(tmp_path):
    _make_gpu_runtime(tmp_path)
    assert OP.resolve_providers("auto", ort_module=None, base_dir=tmp_path) == OP.CPU_ONLY


def test_auto_uses_cuda_when_available_and_runtime_ready(tmp_path):
    _make_gpu_runtime(tmp_path)
    ort = _FakeOrt(available=("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert OP.resolve_providers("auto", ort_module=ort, base_dir=tmp_path) == OP.CUDA_THEN_CPU


def test_auto_stays_cpu_when_runtime_missing(tmp_path):
    ort = _FakeOrt(available=("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert OP.resolve_providers("auto", ort_module=ort, base_dir=tmp_path) == OP.CPU_ONLY


def test_cuda_requested_but_provider_absent_falls_back_with_warning(tmp_path, caplog_dbg):
    _make_gpu_runtime(tmp_path)
    ort = _FakeOrt(available=("CPUExecutionProvider",))
    assert OP.resolve_providers("cuda", ort_module=ort, base_dir=tmp_path) == OP.CPU_ONLY
    assert any("onnx_device=cuda" in m for m in caplog_dbg)


def test_invalid_prefer_is_auto(tmp_path):
    _make_gpu_runtime(tmp_path)
    ort = _FakeOrt(available=("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert OP.resolve_providers("garbage", ort_module=ort, base_dir=tmp_path) == OP.CUDA_THEN_CPU
    assert OP.resolve_providers(None, ort_module=ort, base_dir=tmp_path) == OP.CUDA_THEN_CPU


# --- gpu_runtime_ready -------------------------------------------------------

def test_runtime_ready_false_when_absent(tmp_path):
    assert OP.gpu_runtime_ready(tmp_path) is False


def test_runtime_ready_false_on_bad_json(tmp_path):
    d = tmp_path / OP.GPU_RUNTIME_DIRNAME
    d.mkdir()
    (d / "manifest.json").write_text("{ not json", encoding="utf-8")
    assert OP.gpu_runtime_ready(tmp_path) is False


def test_runtime_ready_false_when_a_file_is_missing(tmp_path):
    _make_gpu_runtime(tmp_path, complete=False)
    assert OP.gpu_runtime_ready(tmp_path) is False


def test_runtime_ready_false_on_empty_required(tmp_path):
    d = tmp_path / OP.GPU_RUNTIME_DIRNAME
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"required": []}), encoding="utf-8")
    assert OP.gpu_runtime_ready(tmp_path) is False


def test_runtime_ready_true_when_complete(tmp_path):
    _make_gpu_runtime(tmp_path)
    assert OP.gpu_runtime_ready(tmp_path) is True


# --- make_session ----------------------------------------------------------

def test_make_session_falls_back_to_cpu_on_cuda_failure(tmp_path, caplog_dbg):
    _make_gpu_runtime(tmp_path)
    ort = _FakeOrt(available=("CUDAExecutionProvider", "CPUExecutionProvider"), fail_on=("cuda",))
    sess = OP.make_session("m.onnx", prefer="auto", ort_module=ort, base_dir=tmp_path, label="T")
    assert sess.providers == OP.CPU_ONLY
    assert ort.calls == [OP.CUDA_THEN_CPU, OP.CPU_ONLY]
    assert any("retrying on CPU" in m for m in caplog_dbg)


def test_make_session_reraises_when_cpu_itself_fails(tmp_path):
    ort = _FakeOrt(available=("CPUExecutionProvider",), fail_on=("cpu",))
    with pytest.raises(RuntimeError):
        OP.make_session("m.onnx", prefer="cpu", ort_module=ort, base_dir=tmp_path)


def test_make_session_passes_resolved_providers(tmp_path):
    _make_gpu_runtime(tmp_path)
    ort = _FakeOrt(available=("CUDAExecutionProvider", "CPUExecutionProvider"))
    sess = OP.make_session("m.onnx", prefer="auto", ort_module=ort, base_dir=tmp_path)
    assert ort.calls == [OP.CUDA_THEN_CPU]
    assert sess.providers == OP.CUDA_THEN_CPU


def test_make_session_cpu_path_is_unchanged(tmp_path):
    ort = _FakeOrt(available=("CPUExecutionProvider", "CUDAExecutionProvider"))
    sess = OP.make_session("m.onnx", prefer="auto", ort_module=ort, base_dir=tmp_path,
                           sess_options="SO")
    assert ort.calls == [OP.CPU_ONLY]
    assert sess.sess_options == "SO"


def test_make_session_without_ort_raises(tmp_path):
    with pytest.raises(ImportError):
        OP.make_session("m.onnx", ort_module=None, base_dir=tmp_path)


# --- preload_gpu_dlls ------------------------------------------------------

def test_preload_noop_without_runtime(tmp_path):
    assert OP.preload_gpu_dlls(base_dir=tmp_path, ort_module=None) is False


def test_preload_invokes_ort_preload_when_ready(tmp_path, monkeypatch):
    _make_gpu_runtime(tmp_path)
    seen = {}

    class _Ort:
        def preload_dlls(self, cuda=False, cudnn=False, directory=None):
            seen.update(cuda=cuda, cudnn=cudnn, directory=directory)

    monkeypatch.setattr(OP, "log_dbg", lambda *a, **k: None)
    assert OP.preload_gpu_dlls(base_dir=tmp_path, ort_module=_Ort()) is True
    assert seen["cuda"] is True and seen["cudnn"] is True
    assert seen["directory"] == str(tmp_path / OP.GPU_RUNTIME_DIRNAME)


# --- app_settings: onnx_device / gpu_setup_prompt --------------------------

def test_settings_defaults():
    s = A.load_settings(A.get_default_config())
    assert s.behavior.onnx_device == "auto"
    assert s.behavior.gpu_setup_prompt == "ask"


def test_settings_old_config_without_keys_falls_back():
    cfg = configparser.ConfigParser()
    cfg.read_dict({"Behavior": {"enable_solo_character_limit": "True",
                                "convert_underscore_to_space": "True",
                                "existing_file_mode": "ASK"}})
    s = A.load_settings(cfg)
    assert s.behavior.onnx_device == "auto"
    assert s.behavior.gpu_setup_prompt == "ask"


def test_settings_invalid_values_and_case_insensitive():
    cfg = A.get_default_config()
    cfg.set("Behavior", "onnx_device", "bogus")
    cfg.set("Behavior", "gpu_setup_prompt", "maybe")
    s = A.load_settings(cfg)
    assert s.behavior.onnx_device == "auto"
    assert s.behavior.gpu_setup_prompt == "ask"
    assert A.parse_onnx_device("  CUDA ") == "cuda"
    assert A.parse_gpu_setup_prompt("DISMISSED") == "dismissed"


def test_settings_round_trip(monkeypatch):
    tmp = Path(tempfile.mkdtemp()) / "config.ini"
    monkeypatch.setattr(A, "CONFIG_PATH", tmp)

    s = A.load_settings(A.get_default_config())
    s.behavior.onnx_device = "cuda"
    s.behavior.gpu_setup_prompt = "dismissed"
    assert A.save_config(s) is True

    written = configparser.ConfigParser()
    written.read(tmp, encoding="utf-8")
    assert written.get("Behavior", "onnx_device") == "cuda"
    assert written.get("Behavior", "gpu_setup_prompt") == "dismissed"

    s2 = A.load_settings(written)
    assert s2.behavior.onnx_device == "cuda"
    assert s2.behavior.gpu_setup_prompt == "dismissed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
