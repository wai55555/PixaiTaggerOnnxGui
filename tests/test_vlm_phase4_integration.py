"""VLM Phase 4: MainWindow integration points (offscreen).

Verifies the "Use VLM connection" toggle and worker routing without exercising
the full model lifecycle.
Run:  QT_QPA_PLATFORM=offscreen python tests/test_vlm_phase4_integration.py
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# isolate config.ini so MainWindow.closeEvent's save doesn't pollute the real one
import app_settings as _A
_A.CONFIG_PATH = Path(tempfile.mkdtemp()) / "config.ini"
import constants as _C
_C.CONFIG_PATH = _A.CONFIG_PATH
import vlm_config as _VC
_vcdir = Path(tempfile.mkdtemp())
_VC.VLM_CONNECTIONS_PATH = _vcdir / "vlm_connections.json"
_VC.VLM_PROFILES_PATH = _vcdir / "vlm_profiles.json"

from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])


def _mw_ready(vlm_enabled: bool = False):
    import main_window
    w = main_window.MainWindow()
    _APP.processEvents()
    _APP.processEvents()  # let QTimer.singleShot(0, initial_load) run
    w._on_use_vlm_toggled(vlm_enabled)   # force a deterministic UI refresh
    _APP.processEvents()
    return w


def test_use_vlm_toggle_surfaces_ui_and_hides_tag_grid():
    w = _mw_ready()
    # off: tag grid shown, VLM buttons hidden (default model is a tagger)
    assert not w.tag_grid_container.isHidden()
    assert w.vlm_settings_button.isHidden() and w.vlm_single_test_button.isHidden()
    assert w.caption_text_edit.isHidden()

    w.use_vlm_check.setChecked(True)   # emits toggled -> _on_use_vlm_toggled
    _APP.processEvents()
    assert w.settings.vlm.enabled is True
    assert not w.vlm_settings_button.isHidden()
    assert not w.vlm_single_test_button.isHidden()
    assert not w.caption_text_edit.isHidden(), "text editor should show for VLM output"
    assert w.tag_grid_container.isHidden()
    assert w.task_combo.isHidden(), "Florence-2 task combo is irrelevant for VLM"
    assert not w.caption_placement_widget.isHidden()

    w.use_vlm_check.setChecked(False)
    _APP.processEvents()
    assert w.settings.vlm.enabled is False
    assert not w.tag_grid_container.isHidden()
    w.close()
    print("  Use-VLM toggle surfaces VLM UI, hides tag grid, restores on off: OK")


def test_model_combo_greyed_while_vlm_on():
    w = _mw_ready(vlm_enabled=False)
    assert w.model_combo.isEnabled()
    w.use_vlm_check.setChecked(True)
    _APP.processEvents()
    assert not w.model_combo.isEnabled(), "model list must be disabled while VLM is on"
    w.use_vlm_check.setChecked(False)
    _APP.processEvents()
    assert w.model_combo.isEnabled(), "unchecking VLM restores the model list"
    w.close()
    print("  model combo greys out while VLM on, restores on uncheck: OK")


def test_start_thread_routes_to_vlm_worker_when_enabled():
    import main_window
    from vlm_worker import VlmCaptionWorker
    from workers import TaggerThreadWorker

    w = main_window.MainWindow()
    _APP.processEvents()
    entry = MagicMock()
    entry.model_type = "tagger"
    entry.model_id = "fake-tagger"
    w._current_model_entry = lambda: entry
    w._update_ui_for_processing = lambda *a, **k: None
    w._check_model_status_and_update_ui = lambda *a, **k: None
    w._make_decision_requester = lambda: None
    w.image_list.clear()

    w.settings.vlm.enabled = False
    w._start_tagging_thread()
    assert isinstance(w._tagger_worker, TaggerThreadWorker), type(w._tagger_worker)
    w._tagger_worker.stop(); w._tagger_thread.quit(); w._tagger_thread.wait(3000)
    w._cleanup_tagger_thread()

    w.settings.vlm.enabled = True
    w._start_tagging_thread()
    assert isinstance(w._tagger_worker, VlmCaptionWorker), type(w._tagger_worker)
    w._tagger_worker.stop(); w._tagger_thread.quit(); w._tagger_thread.wait(3000)
    w._cleanup_tagger_thread()
    w.close()
    print("  _start_tagging_thread: vlm.enabled False->Tagger, True->VlmCaptionWorker (tagger model): OK")


def test_run_button_skips_model_download_when_vlm_enabled():
    import main_window
    w = main_window.MainWindow()
    _APP.processEvents()
    w._is_model_available = lambda: False   # pretend nothing is downloaded
    started = {"tag": 0, "dl": 0}
    w._start_tagging_thread = lambda: started.__setitem__("tag", started["tag"] + 1)
    w._start_download_thread = lambda: started.__setitem__("dl", started["dl"] + 1)
    w._is_downloading = False
    w._tagger_thread = None

    w.settings.vlm.enabled = False
    w.toggle_download_or_start_tagging()
    assert started == {"tag": 0, "dl": 1}, started   # no model -> download

    w.settings.vlm.enabled = True
    w.toggle_download_or_start_tagging()
    assert started == {"tag": 1, "dl": 1}, started   # VLM -> run anyway, no download
    w.close()
    print("  run button: VLM enabled skips the local-model download prompt: OK")


def test_api_key_dialog_verify_and_save():
    import time
    import vlm_diagnostics as D
    from vlm_diagnostics import DiagReport, DiagStatus
    import vlm_secrets
    import api_key_dialog as AKD
    from api_key_dialog import ApiKeyDialog
    from vlm_connections import VlmConnection, ConnectionKind, AuthSpec

    _orig = (AKD.QMessageBox.information, vlm_secrets.set_secret, vlm_secrets.get_secret,
             vlm_secrets.keyring_available, D.diagnose, vlm_secrets.secret_status)
    AKD.QMessageBox.information = staticmethod(lambda *a, **k: None)  # don't block on confirm
    vlm_secrets.secret_status = lambda ref: "missing"

    stored = {}
    vlm_secrets.set_secret = lambda ref, val, persist: (stored.__setitem__(ref, val) or True)
    vlm_secrets.keyring_available = lambda: True
    conn = VlmConnection("builtin-gemini", "Gemini API", ConnectionKind.BUILTIN,
                         "gemini_generate_content", "https://x/v1beta", "m", provider_id="gemini",
                         auth=AuthSpec(type="header_key", secret_ref="vlm/gemini/api_key",
                                       header_name="x-goog-api-key"))
    T2 = lambda sec, key, **kw: key

    def _run(diag_report, key):
        D.diagnose = lambda c, k, do_live_request=True: diag_report
        d = ApiKeyDialog(T2, display_name="Gemini API", secret_ref="vlm/gemini/api_key",
                         conn=conn, key_url="http://k", login_url="http://l", instructions="a\\nb")
        d.key_edit.setText(key)
        d._check_and_save()
        for _ in range(300):
            _APP.processEvents()
            time.sleep(0.01)
            if d._check_thread is None:
                break
        return d

    good = DiagReport("builtin-gemini")
    good.add("Auth", DiagStatus.PASS, "ok")
    good.add("HTTP response", DiagStatus.PASS, "200 OK")
    good.http_status = 200
    d = _run(good, "GOODKEY")
    assert stored.get("vlm/gemini/api_key") == "GOODKEY" and d.saved() is True

    stored.clear()
    bad = DiagReport("builtin-gemini")
    bad.add("Auth", DiagStatus.FAIL, "rejected by the server (401)")
    bad.add("HTTP response", DiagStatus.FAIL, "401 auth rejected")
    bad.http_status = 401
    d2 = _run(bad, "BADKEY")
    assert "vlm/gemini/api_key" not in stored and d2.saved() is False
    d2.close()

    # key accepted but the model id is wrong (400/404) -> save the key, warn, still close
    stored.clear()
    modelwrong = DiagReport("builtin-gemini")
    modelwrong.add("Auth", DiagStatus.PASS, "accepted (server responded 404)")
    modelwrong.add("HTTP response", DiagStatus.FAIL, "404 model / request rejected (auth was accepted)")
    modelwrong.http_status = 404
    d2b = _run(modelwrong, "KEYOK_MODELBAD")
    assert stored.get("vlm/gemini/api_key") == "KEYOK_MODELBAD" and d2b.saved() is True
    assert d2b._model_warning

    stored.clear()
    offline = DiagReport("builtin-gemini")
    offline.add("DNS / TCP", DiagStatus.FAIL, "cannot resolve host")
    d3 = _run(offline, "SOMEKEY")
    assert "vlm/gemini/api_key" not in stored and d3.saved() is False, "unreachable != verified"
    d3.close()

    # a key already registered -> the dialog says so (value stays hidden), no more empty-looking field
    tr = lambda sec, key, **kw: key   # returns the locale key name so we can assert on it
    vlm_secrets.secret_status = lambda ref: "keyring"
    d4 = ApiKeyDialog(tr, display_name="Gemini API", secret_ref="vlm/gemini/api_key",
                      conn=conn, key_url="http://k", login_url="http://l", instructions="x")
    assert d4.current_label.text() == "ApiKey_Current_Registered"
    assert d4.key_edit.text() == "" and d4.key_edit.placeholderText() == "ApiKey_Paste_Placeholder_Update"
    d4.close()
    vlm_secrets.secret_status = lambda ref: "missing"
    d5 = ApiKeyDialog(tr, display_name="Gemini API", secret_ref="vlm/gemini/api_key",
                      conn=conn, key_url="http://k", login_url="http://l", instructions="x")
    assert d5.current_label.text() == "ApiKey_Current_None"
    assert d5.key_edit.placeholderText() == "ApiKey_Paste_Placeholder"
    d5.close()

    # Cloudflare keeps Account ID in this dialog only and requires a full image response.
    cf = VlmConnection(
        "builtin-cloudflare", "Cloudflare Workers AI", ConnectionKind.BUILTIN,
        "openai_chat_completions",
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "@cf/google/gemma-4-26b-a4b-it", provider_id="cloudflare",
        auth=AuthSpec(type="bearer", secret_ref="vlm/cloudflare/api_token"),
    )
    vlm_secrets.secret_status = lambda ref: "missing"
    cf_ids = []
    d6 = ApiKeyDialog(
        tr, display_name="Cloudflare Workers AI", secret_ref="vlm/cloudflare/api_token",
        conn=cf, key_url="http://k", login_url="http://l", instructions="x",
        on_cloudflare_verified=cf_ids.append,
    )
    assert d6.account_id_edit is not None
    d6.key_edit.setText("CFKEY")
    d6._check_and_save()
    assert d6._check_thread is None and d6.status.text() == "ApiKey_Cf_Account_Empty"
    d6.account_id_edit.setText("not-an-account-id")
    d6._check_and_save()
    assert d6._check_thread is None and d6.status.text() == "ApiKey_Cf_Account_Invalid"

    token_only = DiagReport("builtin-cloudflare")
    token_only.add("Auth", DiagStatus.PASS, "token valid")
    token_only.add("HTTP response", DiagStatus.PASS, "token valid and active")
    token_only.add("Caption extraction", DiagStatus.SKIP, "token verify only")
    token_only.http_status = 200
    D.diagnose = lambda c, k, do_live_request=True: token_only
    account_id = "0123456789abcdef0123456789abcdef"
    d6.account_id_edit.setText(account_id)
    d6._check_and_save()
    for _ in range(300):
        _APP.processEvents()
        time.sleep(0.01)
        if d6._check_thread is None:
            break
    assert d6.saved() is False and "vlm/cloudflare/api_token" not in stored
    assert cf_ids == [], "token verification alone must not save the Account ID"

    cf_good = DiagReport("builtin-cloudflare")
    cf_good.add("Auth", DiagStatus.PASS, "ok")
    cf_good.add("HTTP response", DiagStatus.PASS, "200 OK")
    cf_good.add("Caption extraction", DiagStatus.PASS, "got text")
    cf_good.http_status = 200
    checked_urls = []
    D.diagnose = lambda c, k, do_live_request=True: (checked_urls.append(c.base_url) or cf_good)
    d6.account_id_edit.setText(account_id)
    d6._check_and_save()
    for _ in range(300):
        _APP.processEvents()
        time.sleep(0.01)
        if d6._check_thread is None:
            break
    assert stored["vlm/cloudflare/api_token"] == "CFKEY"
    assert cf_ids == [account_id] and d6.saved() is True
    assert checked_urls == [f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"]

    # A registered token can validate a newly entered Account ID without being pasted again.
    stored.clear()
    cf_ids.clear()
    vlm_secrets.secret_status = lambda ref: "keyring"
    vlm_secrets.get_secret = lambda ref: "EXISTING_CF_KEY"
    d7 = ApiKeyDialog(
        tr, display_name="Cloudflare Workers AI", secret_ref="vlm/cloudflare/api_token",
        conn=cf, key_url="http://k", login_url="http://l", instructions="x",
        cloudflare_account_id=account_id, on_cloudflare_verified=cf_ids.append,
    )
    d7._check_and_save()
    for _ in range(300):
        _APP.processEvents()
        time.sleep(0.01)
        if d7._check_thread is None:
            break
    assert d7.saved() is True and cf_ids == [account_id]
    assert stored == {}, "an existing token should not be copied into another secret tier"

    (AKD.QMessageBox.information, vlm_secrets.set_secret, vlm_secrets.get_secret,
     vlm_secrets.keyring_available, D.diagnose, vlm_secrets.secret_status) = _orig
    print("  ApiKeyDialog: generic key checks + Cloudflare Account ID/full request: OK")


def test_dotenv_loading():
    """A .env next to the app feeds the enabled provider keys via the env-var tier,
    without overwriting anything already set."""
    import importlib
    import os
    import shutil
    import vlm_secrets
    from constants import BASE_DIR
    d = Path(tempfile.mkdtemp())
    (d / ".env").write_text(
        'GEMINI_API_KEY=g_key\n# note\nexport MISTRAL_API_KEY="m_key"\n'
        'HF_TOKEN=hf_key\n'
        'CLOUDFLARE_ACCOUNT_ID=acct-x\n',
        encoding="utf-8")
    cwd = os.getcwd()
    # (d) every env name get_secret() can consult: all _ENV_ALIASES values, the
    # generic PIXAI_* fallbacks for the refs asserted below, plus the
    # CLOUDFLARE_ACCOUNT_ID asserted via os.environ directly.
    env_keys = sorted(
        {n for names in vlm_secrets._ENV_ALIASES.values() for n in names}
        | set(vlm_secrets._env_candidates("vlm/gemini/api_key"))
        | set(vlm_secrets._env_candidates("vlm/mistral/api_key"))
        | {"CLOUDFLARE_ACCOUNT_ID"})
    saved = {k: os.environ.pop(k, None) for k in env_keys}
    os.environ["GROQ_API_KEY"] = "preset"
    # (b)+(c) snapshot module state clobbered below / by the reload.
    saved_session = dict(vlm_secrets._session_store)
    vlm_secrets._session_store.clear()
    saved_probe = vlm_secrets._keyring_probe
    # (e) pin both .env candidates (vlm_secrets.py:30: BASE_DIR/.env + cwd/.env):
    # a real file at either spot would feed real keys and shadow the temp-.env.
    # Move each aside, restore in finally.
    _moved = []
    for _p in (Path(BASE_DIR) / ".env", Path(cwd) / ".env"):
        if _p.is_file() and all(str(_p) != m[0] for m in _moved):
            _bak = str(_p) + ".wave3t6bak"
            shutil.move(str(_p), _bak)
            _moved.append((str(_p), _bak))
    # phase2/phase3 leave a module-global `get_secret = lambda ref: "FAKEKEY"`
    # with no restore; reload recovers the genuine three-tier implementation.
    saved_get_secret = vlm_secrets.get_secret
    os.chdir(d)
    try:
        importlib.reload(vlm_secrets)
        # (a) keyring tier wins over env: force a FALSY answer so the temp-.env
        # values show through even with a REAL key in Credential Manager.
        # A truthy dummy stub is FORBIDDEN (it would shadow env the same way).
        _kr = vlm_secrets.keyring
        _orig_get = _kr.get_password if _kr is not None else None
        if _kr is not None:
            _kr.get_password = lambda *a, **k: None  # type: ignore[method-assign]
        try:
            vlm_secrets._load_dotenv()
            assert vlm_secrets.get_secret("vlm/gemini/api_key") == "g_key"
            assert vlm_secrets.get_secret("vlm/mistral/api_key") == "m_key"      # quotes stripped
            assert vlm_secrets.get_secret("vlm/huggingface/api_token") == "hf_key"
            assert os.environ["CLOUDFLARE_ACCOUNT_ID"] == "acct-x"
            assert os.environ["GROQ_API_KEY"] == "preset"                        # not overwritten
        finally:
            if _kr is not None:
                _kr.get_password = _orig_get  # type: ignore[method-assign]
            vlm_secrets.get_secret = saved_get_secret
            vlm_secrets._session_store.clear()
            vlm_secrets._session_store.update(saved_session)
            vlm_secrets._keyring_probe = saved_probe
            for k in env_keys:
                os.environ.pop(k, None)
                if saved.get(k) is not None:
                    os.environ[k] = saved[k]
    finally:
        os.chdir(cwd)
        for _p, _bak in _moved:
            shutil.move(_bak, _p)
    print("  .env loading: enabled provider aliases, quotes/export handled, preset wins: OK")


def test_lang_files_parse_and_expose_vlm_keys():
    """Multi-line ini values (no indent) make configparser reject the whole file,
    so every string silently falls back to the raw key. Guard against that."""
    import configparser
    root = Path(__file__).resolve().parent.parent
    want = ["Settings_Mode_Builtin", "ApiKey_Steps_Gemini", "ApiKey_Steps_Cloudflare",
            "ApiKey_Steps_HuggingFace", "ApiKey_Cf_Account_Invalid",
            "Opt_Detail_maximum_detail", "Opt_Sentence_5", "Opt_CharName_explicit_only",
            "Opt_Markdown_disabled", "Settings_Language_Fixed_Tooltip"]
    for name in ("lang/en.ini", "lang/ja.ini"):
        c = configparser.ConfigParser()
        c.read(root / name, encoding="utf-8")   # raises ParsingError if malformed
        for k in want:
            assert c.has_option("Vlm", k), f"{name}: missing [Vlm] {k}"
        assert "(" not in c.get("Vlm", "Settings_Mode_Builtin")
        assert "\\n" in c.get("Vlm", "ApiKey_Steps_Gemini"), f"{name}: steps not one line"
    print("  lang/*.ini parse; [Vlm] mode/apikey/option keys present: OK")


def test_keyring_available_needs_a_real_backend():
    """import keyring succeeding is not enough - a 'fail' backend (priority<=0)
    must report unavailable, or the key dialog lies about OS storage."""
    import types
    import vlm_secrets as vs

    class _FailKR:
        priority = -1

        def get_password(self, *a):
            return None

    class _RealKR:
        priority = 5

        def get_password(self, *a):
            return None

    orig_kr, orig_probe = vs.keyring, vs._keyring_probe
    try:
        for backend, expected in ((_FailKR(), False), (_RealKR(), True)):
            vs._keyring_probe = None
            vs.keyring = types.SimpleNamespace(get_keyring=lambda b=backend: b)
            assert vs.keyring_available() is expected, (type(backend).__name__, expected)
        vs._keyring_probe = None
        vs.keyring = None
        assert vs.keyring_available() is False
    finally:
        vs.keyring, vs._keyring_probe = orig_kr, orig_probe
    print("  keyring_available(): fail-backend -> False, real -> True, no module -> False: OK")


def test_batch_completed_builds_undo_for_vlm_changes():
    import main_window
    from tagging_core import FileChange
    w = main_window.MainWindow()
    w.undo_manager.clear()
    changes = [FileChange(path=Path("/tmp/x.txt"), previous_content="old", new_content="new caption",
                          was_append=False, added_tags=())]
    w._on_batch_completed(changes)
    assert w.undo_manager.can_undo()
    w.close()
    print("  _on_batch_completed accepts VLM FileChange list -> undo entry: OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} VLM PHASE 4 INTEGRATION TESTS PASSED")
