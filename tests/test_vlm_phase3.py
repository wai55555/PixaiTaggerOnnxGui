"""VLM Phase 3+ tests: worker existing-file handling, exhaustion break, dialogs.

Offline only. Run:  python tests/test_vlm_phase3.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# isolate config.ini: test_settings_dialog_roundtrip calls VlmSettingsDialog._on_save,
# which persists via save_config(). Without this it overwrites the real config.ini.
import app_settings as _A
_A.CONFIG_PATH = Path(tempfile.mkdtemp()) / "config.ini"
import constants as _C
_C.CONFIG_PATH = _A.CONFIG_PATH
# isolate vlm_connections.json / vlm_profiles.json too (the dialog reads/writes them).
import vlm_config as _VC
_vcdir = Path(tempfile.mkdtemp())
_VC.VLM_CONNECTIONS_PATH = _vcdir / "vlm_connections.json"
_VC.VLM_PROFILES_PATH = _vcdir / "vlm_profiles.json"

from PySide6.QtWidgets import QApplication
from PIL import Image

# QApplication (not QCoreApplication) so the settings-dialog test can build widgets
# in the same process as the worker tests.
_APP = QApplication.instance() or QApplication([])

import vlm_transport as T
from vlm_transport import RawHttpResponse
import vlm_config
import vlm_secrets
import vlm_models as M


def _ok_body(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}


def _setup(tmpdir, existing_mode, placement, existing_txt=None):
    import app_settings as A
    import dataclasses
    s = A.load_settings(A.get_default_config())
    s.paths.input_dir = str(tmpdir)
    s.vlm.enabled = True
    s.behavior.existing_file_mode = existing_mode
    s.caption.placement = placement
    for i in range(2):
        Image.new("RGB", (8, 8)).save(tmpdir / f"i{i}.png")
    if existing_txt is not None:
        (tmpdir / "i0.txt").write_text(existing_txt, encoding="utf-8")

    verified = {pid: dataclasses.replace(b, identity_status=M.ModelIdentityStatus.VERIFIED,
                                         provider_constraint=None)
                for pid, b in M.GEMMA_4_26B_A4B_IT.bindings.items()}
    vlm_config.resolve_model_profile = lambda v: dataclasses.replace(M.GEMMA_4_26B_A4B_IT, bindings=verified)
    vlm_secrets.get_secret = lambda ref: "FAKEKEY"
    return s


def _run(worker):
    logs = []
    worker.log_message.connect(lambda m, c: logs.append((m, c)))
    worker.run_captioning()
    return logs


def test_append_mode_default_placement_coerced():
    """existing_file_mode=APPEND but placement stays default OVERWRITE -> must append, not overwrite."""
    app = _APP
    d = Path(tempfile.mkdtemp())
    s = _setup(d, "APPEND", "OVERWRITE", existing_txt="1girl, solo")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("a natural language description"), "")
    try:
        from vlm_worker import VlmCaptionWorker
        _run(VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else "")))
    finally:
        T.execute_http = old
    out = (d / "i0.txt").read_text(encoding="utf-8")
    assert out == "1girl, solo\na natural language description", repr(out)
    print("  APPEND mode + default OVERWRITE placement -> coerced to append: OK")


def test_single_test_saves_to_txt():
    """The single-image test now writes the .txt (same save path as the batch) and
    reports one FileChange for undo."""
    app = _APP
    d = Path(tempfile.mkdtemp())
    s = _setup(d, "OVERWRITE", "OVERWRITE", existing_txt="old caption")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("fresh vlm caption"), "")
    try:
        from vlm_worker import VlmCaptionWorker
        w = VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else ""),
                             selected_file_path=d / "i0.png", single_test=True)
        changes = []
        w.batch_completed.connect(lambda lst: changes.extend(lst))
        shown = []
        w.single_test_result.connect(lambda cap, c, m: shown.append(cap))
        verified = []
        w.binding_verified.connect(lambda prov, prof: verified.append((prov, prof)))
        w.run_captioning()
    finally:
        T.execute_http = old
    assert (d / "i0.txt").read_text(encoding="utf-8") == "fresh vlm caption"
    assert shown == ["fresh vlm caption"]
    assert len(changes) == 1 and changes[0].previous_content == "old caption"
    # gemini's builtin protocol is gemini_generate_content; the mock returns OpenAI-shaped
    # JSON, so gemini fails to parse and the run succeeds via openrouter.
    assert verified == [("openrouter", "gemma-4-26b-a4b-it")], verified
    # the other image is untouched - single test only touches the selected one
    assert not (d / "i1.txt").exists()
    print("  single test writes the selected .txt + one undo FileChange: OK")


def test_single_test_skip_mode_does_not_write():
    app = _APP
    d = Path(tempfile.mkdtemp())
    s = _setup(d, "SKIP", "OVERWRITE", existing_txt="keep me")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("nope"), "")
    try:
        from vlm_worker import VlmCaptionWorker
        w = VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else ""),
                             selected_file_path=d / "i0.png", single_test=True)
        changes = []
        w.batch_completed.connect(lambda lst: changes.extend(lst))
        w.run_captioning()
    finally:
        T.execute_http = old
    assert (d / "i0.txt").read_text(encoding="utf-8") == "keep me"
    assert changes == []
    print("  single test respects SKIP mode (no write, no undo entry): OK")


def test_skip_mode_leaves_existing():
    app = _APP
    d = Path(tempfile.mkdtemp())
    s = _setup(d, "SKIP", "OVERWRITE", existing_txt="keep me")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("new"), "")
    try:
        from vlm_worker import VlmCaptionWorker
        logs = _run(VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else "")))
    finally:
        T.execute_http = old
    assert (d / "i0.txt").read_text(encoding="utf-8") == "keep me"
    assert (d / "i1.txt").read_text(encoding="utf-8") == "new"
    print("  SKIP mode leaves existing .txt untouched: OK")


def test_all_connections_excluded_breaks_once():
    """Every connection returns 401 -> excluded -> batch stops with a single message, not per-image."""
    app = _APP
    d = Path(tempfile.mkdtemp())
    for i in range(2, 30):
        Image.new("RGB", (8, 8)).save(d / f"i{i}.png")
    s = _setup(d, "OVERWRITE", "OVERWRITE")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(401, {}, {"error": {"message": "bad"}}, "unauthorized")
    try:
        from vlm_worker import VlmCaptionWorker
        logs = _run(VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else "")))
    finally:
        T.execute_http = old
    exhausted = [m for m, c in logs if "All_Connections_Exhausted" in m]
    image_failed = [m for m, c in logs if "Image_Failed" in m]
    assert exhausted, "expected an exhaustion message"
    # first image triggers 3x 401 -> all excluded; subsequent images short-circuit (no per-image error spam)
    assert len(image_failed) <= 1, f"too many per-image error lines: {len(image_failed)}"
    print(f"  all-excluded -> single exhaustion message (image_failed lines: {len(image_failed)}): OK")


def test_settings_dialog_roundtrip():
    app = _APP
    import app_settings as A
    s = A.load_settings(A.get_default_config())
    T2 = lambda sec, key, **kw: key
    from vlm_settings_dialog import VlmSettingsDialog
    dlg = VlmSettingsDialog(s, T2)
    dlg.mode_custom.setChecked(True)
    dlg.fee_paid.setChecked(True)
    dlg.max_tokens.setValue(1500)
    assert dlg.language_combo.currentData() == "en" and not dlg.language_combo.isEnabled()
    for r in dlg._route_rows.values():
        if r["conn"].provider_id == "cloudflare":
            r["paid_ok"].setChecked(True)
    assert not hasattr(dlg, "cf_account_edit")
    assert dlg._route_rows["builtin-huggingface"]["enabled"].isChecked() is False
    for cid in ("builtin-vercel", "builtin-openai", "builtin-anthropic"):
        assert dlg._route_rows[cid]["enabled"].isChecked() is False
        assert dlg._route_rows[cid]["conn"].is_known_free_route is False
    assert "builtin-ovhcloud" not in dlg._route_rows
    dlg._on_cloudflare_verified("fedcba9876543210fedcba9876543210")
    assert dlg.strict_check.isChecked() is False       # default off
    dlg.strict_check.setChecked(True)
    dlg._on_save()
    assert s.vlm.execution_mode == "custom_single"
    assert s.vlm.paid_continuation is True
    assert s.vlm.max_output_tokens == 1500
    assert "cloudflare" in s.vlm.paid_connections
    assert s.vlm.cloudflare_account_id == "fedcba9876543210fedcba9876543210"
    assert "gemma-4-26b-a4b-it:cloudflare" in s.vlm.verified_set()
    assert s.vlm.language == "en"
    assert s.vlm.strict_identity is True
    dlg._on_anthropic_workspace_saved("wrkspc_test123")
    assert s.vlm.anthropic_workspace_id == "wrkspc_test123"
    assert A.load_settings(A.load_config()).vlm.anthropic_workspace_id == "wrkspc_test123"

    import vlm_config
    assert vlm_config.build_router_policy(s.vlm).allow_declared_identity is False
    s.vlm.strict_identity = False
    assert vlm_config.build_router_policy(s.vlm).allow_declared_identity is True

    # Earlier worker tests replace this resolver with a fixed Gemma profile.
    # Restore profile-aware resolution before checking the newly shipped profiles.
    vlm_config.resolve_model_profile = lambda v: next(
        (p for p in vlm_config.all_profiles() if p.profile_id == v.model_profile_id), None)
    gpt_i = dlg.profile_combo.findData("openai-gpt-5.6-luna")
    dlg.profile_combo.setCurrentIndex(gpt_i)
    assert dlg._route_rows["builtin-openai"]["model_edit"].currentText() == "gpt-5.6-luna"
    assert dlg._route_rows["builtin-openai"]["conn"].protocol == "openai_responses"
    assert dlg._route_rows["builtin-openai"]["enabled"].isChecked() is False
    claude_i = dlg.profile_combo.findData("claude-haiku-4-5")
    dlg.profile_combo.setCurrentIndex(claude_i)
    assert dlg._route_rows["builtin-anthropic"]["model_edit"].currentText() == \
        "claude-haiku-4-5-20251001"
    assert dlg._route_rows["builtin-anthropic"]["conn"].protocol == "anthropic_messages"
    print("  settings dialog round-trip (incl. strict_identity <-> allow_declared_identity): OK")


def test_settings_dialog_rejects_non_vlm_model():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    s.vlm.model_profile_id = "qwen3.8-27b"
    old_resolver = vlm_config.resolve_model_profile
    vlm_config.resolve_model_profile = lambda v: next(
        (p for p in vlm_config.all_profiles() if p.profile_id == v.model_profile_id), None)
    dlg = None
    try:
        dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
        row = dlg._route_rows["builtin-groq"]
        assert row["enabled"].isEnabled()

        row["model_edit"].setCurrentText("groq/compound-mini")
        dlg._on_model_id_edited("builtin-groq")
        assert row["model_edit"].currentText() == "qwen3.8-27b"
        assert "qwen3.8-27b:groq" not in s.vlm.model_id_override_map()

        dlg._on_model_list("builtin-groq", [
            "groq/compound-mini", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile",
        ])
        assert row["model_ids"] == ["qwen/qwen3.8-27b"]
    finally:
        if dlg is not None:
            dlg.close()
        vlm_config.resolve_model_profile = old_resolver
    print("  settings dialog rejects non-VLM model IDs and filters model list: OK")


def test_settings_dialog_keeps_unbound_route_discoverable():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    s.vlm.model_profile_id = "gemma-4-26b-a4b-it"
    old_resolver = vlm_config.resolve_model_profile
    vlm_config.resolve_model_profile = lambda v: next(
        (p for p in vlm_config.all_profiles() if p.profile_id == v.model_profile_id), None)
    dlg = None
    try:
        dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
        row = dlg._route_rows["builtin-groq"]
        assert not row["enabled"].isEnabled()
        assert row["model_edit"].isEnabled()
        assert row["list_btn"].isEnabled()
        assert row["diag_btn"].isEnabled()

        dlg._on_model_list("builtin-groq", [
            "groq/compound-mini", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile",
        ])
        assert row["model_ids"] == ["qwen/qwen3.8-27b"]
        row["model_edit"].setCurrentText("qwen/qwen3.8-27b")
        dlg._on_model_id_edited("builtin-groq")
        assert vlm_config.build_connection_map(
            s.vlm, vlm_config.resolve_model_profile(s.vlm)
        )["builtin-groq"].model_id == "qwen/qwen3.8-27b"
    finally:
        if dlg is not None:
            dlg.close()
        vlm_config.resolve_model_profile = old_resolver
    print("  unbound route remains available for VLM discovery and diagnosis: OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} VLM PHASE 3 TESTS PASSED")
