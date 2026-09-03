"""VLM Phase 2 tests: executor retry/failover/exclude, diagnostics static, worker with mock.

Offline only. Run:  python tests/test_vlm_phase2.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vlm_transport as T
from vlm_transport import RawHttpResponse, VlmExecutor
from vlm_errors import VlmAttemptError, VlmErrorReason
from vlm_connections import VlmConnection, ConnectionKind, AuthSpec
from vlm_image import PreparedImage
from vlm_profiles import GenerationProfile


def _conn(cid, protocol="openai_chat_completions"):
    return VlmConnection(cid, cid, ConnectionKind.BUILTIN, protocol,
                         "https://x/v1", "m", provider_id=cid, auth=AuthSpec(type="none"))


def _spec():
    return {
        "image": PreparedImage(b"\xff\xd8\xff", "image/jpeg"),
        "profile": GenerationProfile(),
        "system_prompt": "sys", "user_prompt": "u",
    }


def _ok_body(text="a caption"):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}


class _Responder:
    """execute_http の差し替え。connection_id ごとに応答スクリプトを返す。"""
    def __init__(self, script):
        self.script = script       # {url_substr: [resp, resp, ...]} ; resp は RawHttpResponse | VlmAttemptError
        self.calls = []

    def __call__(self, req, *, connect_timeout, read_timeout, verify_tls=True):
        self.calls.append(req.url)
        for key, seq in self.script.items():
            if key in req.url:
                return seq.pop(0) if len(seq) > 1 else seq[0]
        return RawHttpResponse(200, {}, _ok_body(), "")


def _patch(monkey):
    old = T.execute_http
    T.execute_http = monkey
    return old


def test_success_first_connection():
    conns = {"a": _conn("a"), "b": _conn("b")}
    r = _Responder({"x/v1": [RawHttpResponse(200, {}, _ok_body("first"), "")]})
    old = _patch(r)
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.text == "first" and res.connection_id == "a"
        assert len(r.calls) == 1
    finally:
        T.execute_http = old
    print("  success on first connection: OK")


def test_429_failover():
    conns = {"a": _conn("a"), "b": _conn("b")}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        # both connections share a base_url; distinguish by call order.
        responder.n += 1
        if responder.n == 1:
            return RawHttpResponse(429, {"Retry-After": "30"}, {"error": {"code": "rate_limit"}}, "rate limited")
        return RawHttpResponse(200, {}, _ok_body("from b"), "")
    responder.n = 0
    old = _patch(responder)
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.text == "from b" and res.connection_id == "b"
        assert ex.runtime("a").rate_limit is not None and ex.runtime("a").rate_limit.in_cooldown()
        # cooldown carries to next image
        live = ex.live_candidates(["a", "b"])
        assert live == ["b"]
    finally:
        T.execute_http = old
    print("  429 -> failover + cooldown carry-over: OK")


def test_timeout_retry_then_failover():
    conns = {"a": _conn("a"), "b": _conn("b")}
    seq = {"a": [VlmAttemptError(VlmErrorReason.TIMEOUT, None, "t1"),
                 VlmAttemptError(VlmErrorReason.TIMEOUT, None, "t2")]}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        # first two calls (connection a) time out, then b succeeds
        if responder.n < 2:
            responder.n += 1
            return VlmAttemptError(VlmErrorReason.TIMEOUT, None, f"timeout {responder.n}")
        return RawHttpResponse(200, {}, _ok_body("b ok"), "")
    responder.n = 0
    old = _patch(responder)
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.connection_id == "b", res.connection_id
        assert responder.n == 2  # one retry_same on 'a', then failover
        reasons = [a.error_reason for a in res.attempts]
        assert reasons == ["timeout", "timeout"]
    finally:
        T.execute_http = old
    print("  timeout -> retry_same once -> failover: OK")


def test_auth_error_excludes_connection():
    conns = {"a": _conn("a"), "b": _conn("b")}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        if responder.first:
            responder.first = False
            return RawHttpResponse(401, {}, {"error": {"message": "bad key"}}, "unauthorized")
        return RawHttpResponse(200, {}, _ok_body("b ok"), "")
    responder.first = True
    old = _patch(responder)
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.connection_id == "b"
        assert ex.runtime("a").is_excluded and ex.runtime("a").excluded_reason == "auth_error"
        # excluded stays excluded for the next image
        res2 = ex.caption_one(_spec(), ["a", "b"])
        assert res2.connection_id == "b"
    finally:
        T.execute_http = old
    print("  auth error -> exclude connection (persists): OK")


def test_all_fail_returns_error():
    conns = {"a": _conn("a"), "b": _conn("b")}
    old = _patch(lambda req, **kw: RawHttpResponse(503, {}, {}, "server error"))
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert not res.ok and res.error is not None
        assert res.error.reason in (VlmErrorReason.SERVER_ERROR, VlmErrorReason.BAD_RESPONSE, VlmErrorReason.UNKNOWN)
    finally:
        T.execute_http = old
    print("  all candidates fail -> ImageResult.error set: OK")


def test_transport_applies_header_and_query_auth():
    from vlm_connections import AuthSpec
    seen = {}

    def responder(req, **kw):
        seen["headers"] = dict(req.headers)
        seen["params"] = dict(req.params)
        return RawHttpResponse(200, {}, _ok_body("ok"), "")

    # header_key auth -> key goes into the configured header, not Authorization
    c1 = VlmConnection("h", "h", ConnectionKind.CUSTOM_LOCAL, "openai_chat_completions",
                       "http://x/v1", "m", auth=AuthSpec(type="header_key", secret_ref="r",
                                                         header_name="X-Api-Key"))
    old = _patch(responder)
    try:
        ex = VlmExecutor({"h": c1}, lambda ref: "SECRET")
        ex.caption_one(_spec(), ["h"])
        assert seen["headers"].get("X-Api-Key") == "SECRET"
        assert "Authorization" not in seen["headers"]

        # query_key auth -> key goes into the query string
        c2 = VlmConnection("q", "q", ConnectionKind.CUSTOM_LOCAL, "openai_chat_completions",
                           "http://x/v1", "m", auth=AuthSpec(type="query_key", secret_ref="r",
                                                             query_param="api_key"))
        ex2 = VlmExecutor({"q": c2}, lambda ref: "SECRET")
        ex2.caption_one(_spec(), ["q"])
        assert seen["params"].get("api_key") == "SECRET"

        # bearer auth -> Authorization: Bearer
        c3 = VlmConnection("b", "b", ConnectionKind.CUSTOM_EXTERNAL, "openai_chat_completions",
                           "http://x/v1", "m", auth=AuthSpec(type="bearer", secret_ref="r"),
                           request_headers={"anthropic-workspace-id": "wrkspc_test"})
        ex3 = VlmExecutor({"b": c3}, lambda ref: "SECRET")
        ex3.caption_one(_spec(), ["b"])
        assert seen["headers"].get("Authorization") == "Bearer SECRET"
        assert seen["headers"].get("anthropic-workspace-id") == "wrkspc_test"
    finally:
        T.execute_http = old
    print("  transport auth routing (header_key / query_key / bearer): OK")


def test_stop_job_on_prompt_format_error():
    conns = {"a": _conn("a"), "b": _conn("b")}
    # 400 with a prompt/format style error -> BAD_RESPONSE -> failover normally.
    # A dedicated PROMPT_FORMAT_ERROR reason -> stop_job. Simulate via a parser that
    # returns that reason: use a 200 body the OpenAI parser treats as empty is not it;
    # instead inject the reason directly through execute_http returning the error.
    from vlm_errors import VlmAttemptError, VlmErrorReason
    old = _patch(lambda req, **kw: VlmAttemptError(VlmErrorReason.PROMPT_FORMAT_ERROR, 400, "bad prompt shape"))
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.stop_job and not res.ok
        # only the first connection was tried (no failover on stop_job)
        assert len(res.attempts) == 1
    finally:
        T.execute_http = old
    print("  prompt-format error -> stop_job (no failover): OK")


def test_stop_mid_flight():
    conns = {"a": _conn("a"), "b": _conn("b")}
    stopped = {"v": False}
    old = _patch(lambda req, **kw: RawHttpResponse(503, {}, {}, "err"))
    try:
        ex = VlmExecutor(conns, lambda ref: None, stop_checker=lambda: stopped["v"])
        stopped["v"] = True
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.stopped
    finally:
        T.execute_http = old
    print("  stop checker aborts caption_one: OK")


def test_diagnostics_static():
    import vlm_diagnostics as D
    good = _conn("g")
    rep = D.diagnose(good, api_key=None, do_live_request=False)
    names = {i.name: i.status for i in rep.items}
    assert names["URL format"] is D.DiagStatus.PASS
    assert names["Model ID"] is D.DiagStatus.PASS
    assert names["HTTP response"] is D.DiagStatus.SKIP

    bad = VlmConnection("b", "b", ConnectionKind.CUSTOM_EXTERNAL, "openai_chat_completions",
                        "not-a-url", "m", auth=AuthSpec(type="bearer", secret_ref="x"))
    rep2 = D.diagnose(bad, api_key=None, do_live_request=False)
    assert rep2.overall is D.DiagStatus.FAIL

    # a non-cloudflare template hole -> single URL-format FAIL, stop before any probe
    tmpl = VlmConnection("t", "t", ConnectionKind.CUSTOM_EXTERNAL, "openai_chat_completions",
                         "https://x.example/{region}/v1", "m",
                         auth=AuthSpec(type="bearer", secret_ref="x"))
    rep3 = D.diagnose(tmpl, api_key="k", do_live_request=True)
    assert [i.name for i in rep3.items] == ["URL format"]
    assert rep3.items[0].status is D.DiagStatus.FAIL and "region" in rep3.items[0].detail

    for protocol in ("openai_responses", "anthropic_messages"):
        direct = _conn(f"direct-{protocol}", protocol)
        rep4 = D.diagnose(direct, api_key=None, do_live_request=False)
        assert rep4.item("Protocol").status is D.DiagStatus.PASS
    print("  diagnostics static checks: OK")


def test_diagnostics_worker_surfaces_internal_error():
    import vlm_diagnostics as D
    from vlm_worker import VlmDiagnosticsWorker
    old = D.diagnose
    reports = []
    D.diagnose = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe exploded"))
    try:
        worker = VlmDiagnosticsWorker(_conn("broken"), None)
        worker.report_ready.connect(reports.append)
        worker.run()
    finally:
        D.diagnose = old
    assert len(reports) == 1
    item = reports[0].item("Internal diagnostic error")
    assert item is not None and item.status is D.DiagStatus.FAIL
    assert "probe exploded" in item.detail
    print("  diagnostics worker exception -> visible failure report: OK")


def test_diagnostics_cloudflare_token_verify(monkeypatch):
    import vlm_diagnostics as D
    from vlm_transport import RawHttpResponse

    cf = VlmConnection("builtin-cloudflare", "Cloudflare", ConnectionKind.BUILTIN,
                       "openai_chat_completions",
                       "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
                       "@cf/x", auth=AuthSpec(type="bearer", secret_ref="x"),
                       provider_id="cloudflare")

    # valid + active token -> Auth PASS, HTTP response PASS, account-id only a WARN
    monkeypatch.setattr(D, "execute_http", lambda *a, **k: RawHttpResponse(
        200, {}, {"success": True, "result": {"status": "active"}}, ""))
    rep = D.diagnose(cf, api_key="cfut_ok", do_live_request=True)
    names = {i.name: i for i in rep.items}
    assert names["URL format"].status is D.DiagStatus.WARN
    assert names["Auth"].status is D.DiagStatus.PASS
    assert names["HTTP response"].status is D.DiagStatus.PASS
    assert rep.http_status == 200
    assert "Request build" not in names   # generation probe skipped for cloudflare

    # rejected token -> Auth FAIL, http_status 401 (api_key_dialog treats as "key wrong")
    monkeypatch.setattr(D, "execute_http", lambda *a, **k: RawHttpResponse(
        401, {}, {"success": False, "errors": [{"message": "Invalid API Token"}]}, ""))
    rep2 = D.diagnose(cf, api_key="cfut_bad", do_live_request=True)
    n2 = {i.name: i for i in rep2.items}
    assert n2["Auth"].status is D.DiagStatus.FAIL
    assert n2["HTTP response"].status is D.DiagStatus.FAIL
    assert rep2.http_status == 401

    # Account ID が埋まっていれば token/verify で終わらず、実際の画像生成・抽出まで進む。
    cf.base_url = "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1"
    monkeypatch.setattr(D, "execute_http", lambda *a, **k: RawHttpResponse(
        200, {}, {"choices": [{"message": {"content": "cloudflare caption"}}]}, ""))
    rep3 = D.diagnose(cf, api_key="cfut_ok", do_live_request=True)
    n3 = {i.name: i for i in rep3.items}
    assert n3["Request build"].status is D.DiagStatus.PASS
    assert n3["HTTP response"].status is D.DiagStatus.PASS
    assert n3["Caption extraction"].status is D.DiagStatus.PASS
    print("  cloudflare: missing Account ID verifies token only; configured route generates: OK")


def test_diagnostics_live_extraction_branches():
    import vlm_diagnostics as D
    from vlm_protocols import get_protocol
    proto = get_protocol("gemini_generate_content")

    def _cls(body, text="{}"):
        return D._classify_extraction(RawHttpResponse(200, {}, body, text), proto)

    st, _ = _cls({"candidates": [{"content": {"parts": [{"text": "a cat"}]}, "finishReason": "STOP"}]})
    assert st is D.DiagStatus.PASS

    # low token cap -> no parts, finishReason MAX_TOKENS: endpoint is fine -> WARN not FAIL
    st, _ = _cls({"candidates": [{"finishReason": "MAX_TOKENS"}], "usageMetadata": {}})
    assert st is D.DiagStatus.WARN

    cf_error = RawHttpResponse(403, {}, {
        "success": False, "errors": [{"code": 10000, "message": "Authentication error"}]}, "")
    assert D._response_error_detail(cf_error) == "Authentication error"

    assert D.is_billing_or_credit_block(
        "AI Gateway requires a valid credit card on file to service requests")
    assert D.is_billing_or_credit_block(
        "Your credit balance is too low to access the Anthropic API")
    assert D.is_billing_or_credit_block("You have no credits remaining")
    assert not D.is_billing_or_credit_block("Invalid API key")

    # genuinely wrong shape -> FAIL with a body preview
    st, detail = _cls({"unexpected": "shape"}, '{"unexpected": "shape"}')
    assert st is D.DiagStatus.FAIL and "unexpected" in detail

    # non-200 -> SKIP (nothing to extract)
    st, _ = D._classify_extraction(RawHttpResponse(500, {}, {}, "boom"), proto)
    assert st is D.DiagStatus.SKIP

    responses = get_protocol("openai_responses")
    st, _ = D._classify_extraction(RawHttpResponse(200, {}, {
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": []}, "{}"), responses)
    assert st is D.DiagStatus.WARN

    anthropic = get_protocol("anthropic_messages")
    st, _ = D._classify_extraction(RawHttpResponse(200, {}, {
        "content": [], "stop_reason": "max_tokens"}, "{}"), anthropic)
    assert st is D.DiagStatus.WARN
    print("  diagnostics extraction: text->PASS, MAX_TOKENS->WARN, bad shape->FAIL+preview, 500->SKIP: OK")


def test_diagnostics_billing_block_is_not_auth_failure():
    import vlm_diagnostics as D

    conn = VlmConnection(
        "builtin-vercel", "Vercel", ConnectionKind.BUILTIN,
        "openai_chat_completions", "http://localhost/v1", "google/gemma-4-26b-a4b-it",
        provider_id="vercel", auth=AuthSpec(type="bearer", secret_ref="x"))
    old = D.execute_http
    try:
        D.execute_http = lambda *a, **k: RawHttpResponse(
            403, {}, {"error": {"message":
                "AI Gateway requires a valid credit card on file to service requests"}}, "")
        rep = D.diagnose(conn, api_key="valid-key", do_live_request=True)
    finally:
        D.execute_http = old
    assert rep.http_status == 403
    assert rep.item("Auth").status is D.DiagStatus.PASS
    assert "billing / credits unavailable" in rep.item("Auth").detail
    assert rep.item("HTTP response").status is D.DiagStatus.FAIL
    assert "credit card" in rep.item("HTTP response").detail
    print("  diagnostics: billing/card block remains HTTP FAIL but auth PASS: OK")


def test_model_list_fetch():
    import vlm_model_list as ML
    from vlm_connections import VlmConnection, ConnectionKind, AuthSpec
    oai = VlmConnection("c", "c", ConnectionKind.BUILTIN, "openai_chat_completions",
                        "https://x/v1", "m", auth=AuthSpec(type="bearer", secret_ref="r"))
    gem = VlmConnection("g", "g", ConnectionKind.BUILTIN, "gemini_generate_content",
                        "https://y/v1beta", "m",
                        auth=AuthSpec(type="header_key", secret_ref="r", header_name="x-goog-api-key"))
    old = ML.execute_http
    try:
        ML.execute_http = lambda req, **kw: RawHttpResponse(200, {}, {"data": [{"id": "a"}, {"id": "b"}, {"id": "a"}]}, "")
        assert ML.fetch_model_ids(oai, "k") == ["a", "b"]                      # deduped

        ML.execute_http = lambda req, **kw: RawHttpResponse(200, {}, {"models": [
            {"name": "models/gemma-3-27b-it", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}]}, "")
        assert ML.fetch_model_ids(gem, "k") == ["gemma-3-27b-it"]              # models/ stripped, non-vision filtered

        ML.execute_http = lambda req, **kw: RawHttpResponse(401, {}, {}, "unauthorized")
        assert ML.fetch_model_ids(oai, "bad").reason is VlmErrorReason.AUTH_ERROR

        ML.execute_http = lambda req, **kw: RawHttpResponse(404, {}, None, "nope")
        assert ML.fetch_model_ids(oai, "k").reason is VlmErrorReason.BAD_RESPONSE

        # Cloudflare: different endpoint + {"result":[{"name":...,"task":{"name":...}}]}
        cf = VlmConnection("cf", "cf", ConnectionKind.BUILTIN, "openai_chat_completions",
                           "https://api.cloudflare.com/client/v4/accounts/acct-1/ai/v1", "m",
                           provider_id="cloudflare", auth=AuthSpec(type="bearer", secret_ref="r"))
        seen = {}
        def _cf(req, **kw):
            seen["url"] = req.url
            return RawHttpResponse(200, {}, {"result": [
                {"name": "@cf/meta/llama-3.2-11b-vision", "task": {"name": "Image-to-Text"}},
                {"name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}}]}, "")
        ML.execute_http = _cf
        assert ML.fetch_model_ids(cf, "k") == ["@cf/meta/llama-3.2-11b-vision"]
        assert seen["url"].endswith("/ai/models/search")

        # Anthropic uses the same {data:[{id:...}]} shape but requires a version header.
        anthropic = VlmConnection("an", "an", ConnectionKind.BUILTIN, "anthropic_messages",
                                  "https://api.anthropic.com/v1", "m", provider_id="anthropic",
                                  auth=AuthSpec(type="header_key", secret_ref="r",
                                                header_name="x-api-key"),
                                  request_headers={"anthropic-workspace-id": "wrkspc_test"})
        def _anthropic(req, **kw):
            seen["anthropic_headers"] = dict(req.headers)
            return RawHttpResponse(200, {}, {"data": [{"id": "claude-haiku-4-5-20251001"}]}, "")
        ML.execute_http = _anthropic
        assert ML.fetch_model_ids(anthropic, "ak") == ["claude-haiku-4-5-20251001"]
        assert seen["anthropic_headers"]["anthropic-version"] == "2023-06-01"
        assert seen["anthropic_headers"]["x-api-key"] == "ak"
        assert seen["anthropic_headers"]["anthropic-workspace-id"] == "wrkspc_test"
    finally:
        ML.execute_http = old
    print("  model list fetch: openai dedup, gemini strip+filter, cloudflare search endpoint, 401/404: OK")


def test_worker_batch_with_mock(tmp=None):
    import os, tempfile
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PIL import Image
    app = QCoreApplication.instance() or QCoreApplication([])

    d = Path(tempfile.mkdtemp())
    for i in range(3):
        Image.new("RGB", (8, 8)).save(d / f"i{i}.png")
    (d / "i1.txt").write_text("1girl, solo", encoding="utf-8")

    import app_settings as A
    s = A.load_settings(A.get_default_config())
    s.paths.input_dir = str(d)
    s.vlm.enabled = True
    s.behavior.existing_file_mode = "APPEND"
    s.caption.placement = "APPEND"

    # make all three builtin bindings verified + provide fake auth + mock http
    import vlm_models as M, dataclasses, vlm_secrets, vlm_config
    verified = {pid: dataclasses.replace(b, identity_status=M.ModelIdentityStatus.VERIFIED, provider_constraint=None)
               for pid, b in M.GEMMA_4_26B_A4B_IT.bindings.items()}
    M.GEMMA_4_26B_A4B_IT.__dict__  # frozen; patch registry instead
    orig_reg = vlm_config.default_registry
    vlm_config.resolve_model_profile = lambda v: dataclasses.replace(M.GEMMA_4_26B_A4B_IT, bindings=verified)
    vlm_secrets.get_secret = lambda ref: "FAKEKEY"

    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("a detailed natural language description of the scene"), "")

    from vlm_worker import VlmCaptionWorker
    logs = []
    prog = []
    batch = {"v": None}
    w = VlmCaptionWorker(s, decision_requester=None, get_string=lambda *a, **k: a[-1] if a else "")
    w.log_message.connect(lambda m, c: logs.append((m, c)))
    w.progress_update.connect(lambda a, b: prog.append((a, b)))
    w.batch_completed.connect(lambda lst: batch.__setitem__("v", lst))
    try:
        w.run_captioning()
    finally:
        T.execute_http = old

    txt0 = (d / "i0.txt").read_text(encoding="utf-8")
    txt1 = (d / "i1.txt").read_text(encoding="utf-8")
    assert txt0 == "a detailed natural language description of the scene"
    assert txt1.startswith("1girl, solo\n") and "natural language description" in txt1
    assert batch["v"] is not None and len(batch["v"]) == 3
    assert prog and prog[-1] == (3, 3)
    print(f"  worker batch (mock http): OK  ({len(batch['v'])} files written, {len(prog)} progress)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} VLM PHASE 2 TESTS PASSED")
