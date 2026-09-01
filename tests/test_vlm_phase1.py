"""VLM Phase 0/1 backend unit tests (260901_VLM_spec.md 12.1).

Offline only - no network, no API keys. Run:  python tests/test_vlm_phase1.py
"""
import io
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import vlm_models as M
import vlm_profiles as P
import vlm_image as IMG
import vlm_errors as E
import vlm_protocols as PROTO
import vlm_connections as C
import vlm_persistence as PERS
import vlm_router as R
import vlm_ratelimit as RL


def test_model_alias_and_identity():
    reg = M.default_registry()
    assert reg.resolve_alias("gemma-4-26b-a4b-it") is M.GEMMA_4_26B_A4B_IT
    assert reg.resolve_alias("google/gemma-4-26b-a4b-it:free") is M.GEMMA_4_26B_A4B_IT
    assert reg.resolve_alias("@cf/google/gemma-4-26b-a4b-it") is M.GEMMA_4_26B_A4B_IT
    assert reg.resolve_alias("totally-unknown") is None

    verified = M.ModelBinding("x", "m", M.ModelIdentityStatus.VERIFIED)
    assert verified.is_strict_fallback_eligible()

    # provider_constraint present but not pinned -> VERIFIED downgraded to DECLARED
    loose = M.ModelBinding("x", "m", M.ModelIdentityStatus.VERIFIED,
                           M.ProviderConstraint(allowed_providers=(), allow_fallbacks=True))
    assert loose.effective_identity_status() is M.ModelIdentityStatus.DECLARED
    assert not loose.is_strict_fallback_eligible()

    pinned = M.ModelBinding("x", "m", M.ModelIdentityStatus.VERIFIED,
                            M.ProviderConstraint(allowed_providers=("prov-a",), allow_fallbacks=False))
    assert pinned.is_strict_fallback_eligible()

    assert M.parse_identity_status("garbage") is M.ModelIdentityStatus.UNKNOWN
    assert not M.VlmModelProfile("p", "P", "m", quantization="unknown").quantization_is_strict()
    assert M.GEMMA_4_26B_A4B_IT.quantization_is_strict()
    print("  model alias + identity: OK")


def test_prompt_building():
    base = P.GenerationProfile()
    sp = P.build_system_prompt(base)
    assert "highly detailed English" in sp
    assert "as many sentences as necessary" in sp          # automatic
    assert "no bullet lists" in sp.lower() or "no bullet" in sp.lower()  # markdown disabled
    assert "unambiguously clear" in sp                     # explicit_only

    three = P.GenerationProfile(sentence_mode=P.SentenceMode.S3)
    assert "approximately 3 sentences" in P.build_system_prompt(three)
    assert "exactly" not in P.build_system_prompt(three).lower()

    one = P.GenerationProfile(sentence_mode=P.SentenceMode.S1)
    assert "approximately 1 sentence." in P.build_system_prompt(one)

    md_ok = P.GenerationProfile(markdown=P.MarkdownMode.ALLOWED)
    assert "no bullet" not in P.build_system_prompt(md_ok).lower()

    no_id = P.GenerationProfile(character_name_mode=P.CharacterNameMode.DO_NOT_IDENTIFY)
    assert "Do not state any character name" in P.build_system_prompt(no_id)

    custom = P.GenerationProfile(custom_system_prompt="JUST DO IT")
    assert P.build_system_prompt(custom) == "JUST DO IT"

    # from_mapping clamps / falls back
    g = P.GenerationProfile.from_mapping({"detail_level": "bogus", "max_output_tokens": 999999,
                                          "sentence_mode": "3", "markdown": "allowed"})
    assert g.detail_level is P.DetailLevel.MAXIMUM_DETAIL
    assert g.max_output_tokens == 32768
    assert g.sentence_mode is P.SentenceMode.S3
    assert g.markdown is P.MarkdownMode.ALLOWED
    print("  prompt building: OK")


def test_extract_by_path():
    obj = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 5}}
    assert PROTO.extract_by_path(obj, "choices[0].message.content") == "hello"
    assert PROTO.extract_by_path(obj, "choices[0].finish_reason") == "stop"
    assert PROTO.extract_by_path(obj, "usage.prompt_tokens") == 5
    assert PROTO.extract_by_path(obj, "choices[1].message.content") is None
    assert PROTO.extract_by_path(obj, "choices[0].missing") is None
    assert PROTO.extract_by_path(obj, "") is None
    nested = {"a": {"b": [[{"c": 1}]]}}
    assert PROTO.extract_by_path(nested, "a.b[0][0].c") == 1
    print("  extract_by_path: OK")


def _prepared():
    return IMG.PreparedImage(data=b"\xff\xd8\xff", mime_type="image/jpeg")


def test_protocol_parse():
    op = PROTO.OpenAIChatCompletionsProtocol()
    ok = op.parse_response(200, {"choices": [{"message": {"content": " a caption "},
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 20}}, "")
    assert ok.ok and ok.text == "a caption" and ok.completion_tokens == 20

    empty = op.parse_response(200, {"choices": [{"message": {"content": ""}}]}, "")
    assert not empty.ok and empty.error.reason is E.VlmErrorReason.EMPTY_RESPONSE

    filt = op.parse_response(200, {"choices": [{"finish_reason": "content_filter",
                                               "message": {"content": ""}}]}, "")
    assert filt.error.reason is E.VlmErrorReason.CONTENT_POLICY

    e429 = op.parse_response(429, {"error": {"code": "rate_limit"}}, "too many")
    assert e429.error.reason is E.VlmErrorReason.RATE_LIMITED

    e401 = op.parse_response(401, {}, "bad key")
    assert e401.error.reason is E.VlmErrorReason.AUTH_ERROR

    gm = PROTO.GeminiGenerateContentProtocol()
    gok = gm.parse_response(200, {"candidates": [{"content": {"parts": [{"text": "desc"}]},
                                                  "finishReason": "STOP"}],
                                  "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 7}}, "")
    assert gok.ok and gok.text == "desc" and gok.completion_tokens == 7

    gblock = gm.parse_response(200, {"promptFeedback": {"blockReason": "SAFETY"}}, "")
    assert gblock.error.reason is E.VlmErrorReason.CONTENT_POLICY

    # thinking model: parts[0] is an empty thought, the answer is in a later part
    gthink = gm.parse_response(200, {"candidates": [{"content": {"parts": [
        {"text": "", "thought": True},
        {"text": "a solid grey square"},
    ]}, "finishReason": "STOP"}]}, "")
    assert gthink.ok and gthink.text == "a solid grey square", gthink.text

    # only a thought part, no answer -> still EMPTY_RESPONSE
    gthink_only = gm.parse_response(200, {"candidates": [{"content": {"parts": [
        {"text": "", "thought": True}]}, "finishReason": "MAX_TOKENS"}]}, "")
    assert not gthink_only.ok and gthink_only.error.reason is E.VlmErrorReason.EMPTY_RESPONSE
    print("  protocol parse (openai + gemini, incl. thinking parts): OK")


def test_protocol_build_request():
    spec = PROTO.VlmCallSpec(model_id="m", system_prompt="sys", user_prompt="u",
                             image=_prepared(), profile=P.GenerationProfile(temperature=0.4))
    op = PROTO.OpenAIChatCompletionsProtocol()
    req = op.build_request("http://localhost:1234/v1", "KEY", spec)
    assert req.url.endswith("/v1/chat/completions")
    assert req.headers["Authorization"] == "Bearer KEY"
    assert req.json_body["model"] == "m"
    assert req.json_body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert req.json_body["temperature"] == 0.4
    assert req.json_body["max_tokens"] == 1024

    gm = PROTO.GeminiGenerateContentProtocol()
    greq = gm.build_request("https://x/v1beta", "GKEY", spec)
    assert greq.url == "https://x/v1beta/models/m:generateContent"
    assert greq.headers["x-goog-api-key"] == "GKEY"
    assert greq.json_body["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"

    rendered = PROTO.render_template('{"m":"{{model}}","p":"{{system_prompt}}"}', spec)
    assert rendered == '{"m":"m","p":"sys"}'
    print("  protocol build_request + template: OK")


def test_error_classification():
    def mk(reason):
        return E.VlmAttemptError(reason)
    assert mk(E.VlmErrorReason.TIMEOUT).classify(consecutive_timeouts=1) is E.VlmErrorClass.RETRY_SAME
    assert mk(E.VlmErrorReason.TIMEOUT).classify(consecutive_timeouts=2) is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.TIMEOUT).classify(consecutive_timeouts=1, already_retried_same=True) is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.RATE_LIMITED).classify() is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.SERVER_ERROR).classify() is E.VlmErrorClass.RETRY_SAME
    assert mk(E.VlmErrorReason.SERVER_ERROR).classify(already_retried_same=True) is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.AUTH_ERROR).classify() is E.VlmErrorClass.EXCLUDE
    assert mk(E.VlmErrorReason.MODEL_UNSUPPORTED).classify() is E.VlmErrorClass.EXCLUDE
    assert mk(E.VlmErrorReason.CONTENT_POLICY).classify() is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.PROMPT_FORMAT_ERROR).classify() is E.VlmErrorClass.STOP_JOB
    assert mk(E.VlmErrorReason.EMPTY_RESPONSE).classify() is E.VlmErrorClass.RETRY_SAME
    assert E.reason_from_http_status(429) is E.VlmErrorReason.RATE_LIMITED
    assert E.reason_from_http_status(503) is E.VlmErrorReason.SERVER_ERROR
    assert E.reason_from_http_status(404) is E.VlmErrorReason.MODEL_UNSUPPORTED
    print("  error classification: OK")


def test_newline_join_and_dedup():
    assert PERS.combine_caption("", "cap", "APPEND") == "cap"
    assert PERS.combine_caption("tags", "", "APPEND") == "tags"
    assert PERS.combine_caption("tags", "cap", "APPEND") == "tags\ncap"
    assert PERS.combine_caption("tags", "cap", "PREPEND") == "cap\ntags"
    assert PERS.combine_caption("tags", "cap", "OVERWRITE") == "cap"
    assert PERS.combine_caption("tags", "cap", "bogus") == "cap"  # -> OVERWRITE

    assert PERS.caption_already_present("a\nmy caption\nb", "my caption")
    assert PERS.caption_already_present("my caption", "my caption")
    assert not PERS.caption_already_present("a\nb", "my caption")
    print("  newline join + dedup: OK")


def test_atomic_save():
    d = Path(tempfile.mkdtemp())
    # new file
    p = d / "img1.txt"
    out = PERS.save_caption(p, "brand new caption", "OVERWRITE")
    assert out.written and out.previous_content is None and p.read_text(encoding="utf-8") == "brand new caption"

    # append to existing
    p2 = d / "img2.txt"
    p2.write_text("1girl, solo", encoding="utf-8")
    out2 = PERS.save_caption(p2, "a long natural caption", "APPEND")
    assert out2.written and p2.read_text(encoding="utf-8") == "1girl, solo\na long natural caption"

    # duplicate append -> skipped, file unchanged
    out3 = PERS.save_caption(p2, "a long natural caption", "APPEND")
    assert not out3.written and out3.skipped_reason == "duplicate"
    assert p2.read_text(encoding="utf-8") == "1girl, solo\na long natural caption"

    # overwrite identical -> no_change
    p3 = d / "img3.txt"
    p3.write_text("same", encoding="utf-8")
    out4 = PERS.save_caption(p3, "same", "OVERWRITE")
    assert not out4.written and out4.skipped_reason == "no_change"

    # no leftover temp files
    assert not list(d.glob("*.vlmtmp"))

    # unreadable existing file raises (caller must treat as error, not overwrite)
    p4 = d / "img4.txt"
    p4.write_bytes(b"\xff\xfe\x00bad")
    try:
        PERS.save_caption(p4, "cap", "APPEND")
        assert False, "expected read error"
    except UnicodeDecodeError:
        pass
    print("  atomic save (new/append/dup/no-change/unreadable): OK")


def test_router_builtin_fallback():
    profile = M.GEMMA_4_26B_A4B_IT
    # make all three bindings VERIFIED for this test
    import dataclasses
    verified_bindings = {pid: dataclasses.replace(b, identity_status=M.ModelIdentityStatus.VERIFIED,
                                                  provider_constraint=None)
                         for pid, b in profile.bindings.items()}
    profile = dataclasses.replace(profile, bindings=verified_bindings)

    conns = {c.connection_id: c for c in C.default_builtin_connections()}
    has_auth = {cid: True for cid in conns}

    # free_only: cloudflare is not a known free route -> excluded
    pol = R.RouterPolicy(free_only=True, paid_continuation=False)
    cs = R.select_candidates(profile, conns, pol, has_auth=has_auth)
    assert cs.connection_ids == ["builtin-gemini", "builtin-openrouter"]
    assert cs.excluded.get("builtin-cloudflare") == "fee_policy"

    # paid_continuation + cloudflare allowed -> included last
    conns["builtin-cloudflare"].paid_continuation_allowed = True
    pol2 = R.RouterPolicy(free_only=True, paid_continuation=True)
    cs2 = R.select_candidates(profile, conns, pol2, has_auth=has_auth)
    assert cs2.connection_ids == ["builtin-gemini", "builtin-openrouter", "builtin-cloudflare"]

    # missing auth on gemini -> excluded
    cs3 = R.select_candidates(profile, conns, pol, has_auth={"builtin-openrouter": True})
    assert "builtin-gemini" not in cs3.connection_ids and cs3.excluded["builtin-gemini"] == "no_auth"

    # cooldown on openrouter
    cs4 = R.select_candidates(profile, conns, pol, has_auth=has_auth,
                              cooldown_until={"builtin-openrouter": time.time() + 999})
    assert cs4.connection_ids == ["builtin-gemini"]

    # shipped profile: bindings are DECLARED. Default policy allows DECLARED
    # (user picks the services + order), so gemini/openrouter still qualify.
    cs5 = R.select_candidates(M.GEMMA_4_26B_A4B_IT, conns, pol, has_auth=has_auth)
    assert cs5.connection_ids == ["builtin-gemini", "builtin-openrouter"]

    # strict opt-out: allow_declared_identity=False -> DECLARED excluded as before
    strict = R.RouterPolicy(free_only=True, allow_declared_identity=False)
    cs6 = R.select_candidates(M.GEMMA_4_26B_A4B_IT, conns, strict, has_auth=has_auth)
    assert not cs6.has_candidates and cs6.rejected_reason == "no_verified_free_candidate"
    assert cs6.excluded.get("builtin-gemini") == "not_verified"

    # UNKNOWN identity is always excluded, even with allow_declared_identity=True
    import dataclasses as _dc
    unknown = _dc.replace(M.GEMMA_4_26B_A4B_IT, bindings={
        pid: _dc.replace(b, identity_status=M.ModelIdentityStatus.UNKNOWN, provider_constraint=None)
        for pid, b in M.GEMMA_4_26B_A4B_IT.bindings.items()})
    cs7 = R.select_candidates(unknown, conns, pol, has_auth=has_auth)
    assert not cs7.has_candidates and cs7.excluded.get("builtin-gemini") == "identity_unknown"
    print("  router builtin_fallback: OK")


def test_router_custom_single():
    ext = C.VlmConnection.from_mapping({
        "connection_id": "cust-ext", "display_name": "ext", "kind": "custom_external",
        "protocol": "openai_chat_completions", "base_url": "https://api.example.com/v1",
        "model_id": "m", "auth": {"type": "bearer", "secret_ref": "x"},
    })
    loc = C.VlmConnection.from_mapping({
        "connection_id": "cust-loc", "display_name": "loc", "kind": "custom_local",
        "protocol": "openai_chat_completions", "base_url": "http://127.0.0.1:1234/v1",
        "model_id": "m", "auth": {"type": "none"},
    })
    conns = {c.connection_id: c for c in (ext, loc)}
    profile = M.GEMMA_4_26B_A4B_IT

    # free_only blocks external custom
    pol = R.RouterPolicy(execution_mode=R.ExecutionMode.CUSTOM_SINGLE, free_only=True,
                         selected_connection_id="cust-ext")
    cs = R.select_candidates(profile, conns, pol, has_auth={"cust-ext": True})
    assert not cs.has_candidates and cs.rejected_reason == "free_only_blocks_external_custom"

    # free_only allows local custom (single)
    pol2 = R.RouterPolicy(execution_mode=R.ExecutionMode.CUSTOM_SINGLE, free_only=True,
                          selected_connection_id="cust-loc")
    cs2 = R.select_candidates(profile, conns, pol2)
    assert cs2.connection_ids == ["cust-loc"]

    # free_only OFF -> external custom single allowed (with auth)
    pol3 = R.RouterPolicy(execution_mode=R.ExecutionMode.CUSTOM_SINGLE, free_only=False,
                          selected_connection_id="cust-ext")
    cs3 = R.select_candidates(profile, conns, pol3, has_auth={"cust-ext": True})
    assert cs3.connection_ids == ["cust-ext"]

    # custom is never mixed into builtin fallback
    pol4 = R.RouterPolicy(execution_mode=R.ExecutionMode.BUILTIN_FALLBACK, free_only=True)
    cs4 = R.select_candidates(profile, conns, pol4)
    assert not cs4.has_candidates  # no builtin connections present at all
    print("  router custom_single: OK")


def test_ratelimit():
    now = 1_000_000.0
    st = RL.RateLimitState("c1")
    RL.update_from_429(st, {"Retry-After": "30"}, now=now)
    assert st.cooldown_until_utc == now + 30 and st.source == "header"

    st2 = RL.RateLimitState("c2")
    RL.update_from_429(st2, {"x-ratelimit-reset": str(now + 45)}, now=now)
    assert abs(st2.cooldown_until_utc - (now + 45)) < 1 and st2.source == "header"

    st3 = RL.RateLimitState("c3")
    RL.update_from_429(st3, {}, now=now)
    assert st3.cooldown_until_utc == now + RL.ESTIMATED_COOLDOWN_DEFAULT_S and st3.source == "estimated"

    st4 = RL.RateLimitState("c4")
    RL.update_from_429(st4, {"Retry-After": "99999"}, now=now)
    assert st4.cooldown_until_utc == now + RL.ESTIMATED_COOLDOWN_CAP_S  # capped

    assert st.in_cooldown(now + 10) and not st.in_cooldown(now + 31)
    RL.clear(st)
    assert not st.in_cooldown(now + 10)
    print("  ratelimit cooldown: OK")


def _make_img(w, h, mode="RGB", color=(120, 60, 30)):
    return Image.new(mode, (w, h), color)


def test_image_preprocess():
    # landscape resize to long edge
    prepared = IMG.prepare_image(_make_img(4000, 2000),
                                 IMG.ImagePreprocessConfig(max_long_edge=1000, fmt="jpeg"))
    reopened = Image.open(io.BytesIO(prepared.data))
    assert reopened.size == (1000, 500)
    assert prepared.mime_type == "image/jpeg"
    assert prepared.data_url.startswith("data:image/jpeg;base64,")
    assert len(prepared.base64) > 0

    # small image is not upscaled
    small = IMG.prepare_image(_make_img(100, 80), IMG.ImagePreprocessConfig(max_long_edge=1000))
    assert Image.open(io.BytesIO(small.data)).size == (100, 80)

    # RGBA is flattened onto the configured color, output is RGB
    rgba = _make_img(50, 50, "RGBA", (0, 0, 0, 0))
    flat = IMG.prepare_image(rgba, IMG.ImagePreprocessConfig(fmt="png", flatten_rgba_color=(255, 255, 255)))
    out = Image.open(io.BytesIO(flat.data))
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255)
    assert flat.mime_type == "image/png"

    # bytes input
    buf = io.BytesIO()
    _make_img(30, 30).save(buf, format="PNG")
    b = IMG.prepare_image(buf.getvalue())
    assert b.mime_type in ("image/jpeg", "image/png")
    print("  image preprocess: OK")


def test_connection_locality():
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "http://localhost:1234/v1") is C.ConnectionKind.CUSTOM_LOCAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "http://192.168.1.9:8000") is C.ConnectionKind.CUSTOM_LOCAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "https://api.openai.com/v1") is C.ConnectionKind.CUSTOM_EXTERNAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "") is C.ConnectionKind.CUSTOM_EXTERNAL  # unknown -> external
    assert C.resolve_custom_kind(C.ConnectionLocality.LOCAL, "https://api.openai.com/v1") is C.ConnectionKind.CUSTOM_LOCAL

    ext = C.VlmConnection("id", "n", C.ConnectionKind.CUSTOM_EXTERNAL, "openai_chat_completions", "u", "m")
    loc = C.VlmConnection("id", "n", C.ConnectionKind.CUSTOM_LOCAL, "openai_chat_completions", "u", "m")
    bi_free = C.VlmConnection("id", "n", C.ConnectionKind.BUILTIN, "x", "u", "m", is_known_free_route=True)
    bi_paid = C.VlmConnection("id", "n", C.ConnectionKind.BUILTIN, "x", "u", "m", is_known_free_route=False)
    assert not ext.free_for_automation
    assert loc.free_for_automation
    assert bi_free.free_for_automation
    assert not bi_paid.free_for_automation
    print("  connection locality + free_for_automation: OK")


def test_verified_binding_promotion():
    import types
    import dataclasses
    import vlm_config as CFG

    prof_id = M.GEMMA_4_26B_A4B_IT.profile_id
    vlm = types.SimpleNamespace(model_profile_id=prof_id, verified_bindings="",
                                verified_set=lambda: {t.strip() for t in vlm.verified_bindings.split(",") if t.strip()})

    # mark: adds a token, idempotent, keyed by profile:provider
    assert CFG.mark_binding_verified(vlm, "gemini", profile_id=prof_id) is True
    assert CFG.mark_binding_verified(vlm, "gemini", profile_id=prof_id) is False
    assert vlm.verified_bindings == f"{prof_id}:gemini"

    # resolve_model_profile promotes only the listed binding; others keep shipped status
    prof = CFG.resolve_model_profile(vlm)
    assert prof.bindings["gemini"].identity_status is M.ModelIdentityStatus.VERIFIED
    assert prof.bindings["openrouter"].identity_status is M.ModelIdentityStatus.DECLARED

    # strict router (allow_declared_identity=False) now keeps the promoted gemini, drops the rest
    conns = {c.connection_id: c for c in C.default_builtin_connections()}
    strict = R.RouterPolicy(free_only=True, allow_declared_identity=False)
    cs = R.select_candidates(prof, conns, strict, has_auth={cid: True for cid in conns})
    assert cs.connection_ids == ["builtin-gemini"], cs.connection_ids
    print("  verified binding promotion: mark + resolve promote + strict router: OK")


def test_multi_provider_profiles():
    import types
    import vlm_config as CFG

    ids = {p.profile_id for p in M.default_registry().all_profiles()}
    assert {"gemma-4-26b-a4b-it", "gemma-4-31b-it", "qwen3.8-27b", "qwen3.6-27b", "pixtral-12b"} <= ids

    def _s(profile_id):
        s = types.SimpleNamespace(model_profile_id=profile_id, paid_connections="",
                                  cloudflare_account_id="", verified_bindings="", model_id_overrides="")
        s.verified_set = lambda: set()
        s.order_list = lambda: ["gemini", "openrouter", "cloudflare"]

        def _ov():
            out = {}
            for tok in s.model_id_overrides.split(","):
                if "=" in tok:
                    k, _, v = tok.strip().partition("=")
                    if k.strip() and v.strip():
                        out[k.strip()] = v.strip()
            return out
        s.model_id_override_map = _ov
        return s

    s = _s("qwen3.8-27b")
    prof = CFG.resolve_model_profile(s)
    cm = CFG.build_connection_map(s, prof)
    on = {cid for cid, c in cm.items() if c.enabled and c.kind is C.ConnectionKind.BUILTIN}
    assert on == {"builtin-openrouter", "builtin-nvidia", "builtin-groq"}, on
    assert cm["builtin-openrouter"].model_id == "qwen/qwen3.8-27b"
    assert cm["builtin-gemini"].enabled is False
    # nvidia/groq are appended to the fallback order even though connection_order omits them
    assert CFG.ordered_builtin_provider_ids(s, prof) == \
        ["gemini", "openrouter", "cloudflare", "nvidia", "groq"]

    CFG.set_model_id_override(s, "nvidia", "qwen/qwen3-vl-32b-instruct", profile_id="qwen3.8-27b")
    assert CFG.build_connection_map(s, prof)["builtin-nvidia"].model_id == "qwen/qwen3-vl-32b-instruct"
    CFG.set_model_id_override(s, "nvidia", "", profile_id="qwen3.8-27b")
    assert "qwen3.8-27b:nvidia" not in s.model_id_override_map()
    print("  multi-provider profiles: registry, profile-scoped connection map, model-id override: OK")


def test_model_id_match_against_profile():
    reg = M.default_registry()
    gemma = reg.get("gemma-4-26b-a4b-it")
    qwen = reg.get("qwen3.8-27b")

    # exact alias in the list -> score 1.0
    bid, sc = M.match_model_id(gemma, "openrouter",
                               ["google/gemma-4-26b-a4b-it:free", "qwen/qwen-2.5-vl-72b", "gpt-4o"])
    assert bid == "google/gemma-4-26b-a4b-it:free" and sc == 1.0

    # a different size (27b vs 26b) is NOT the same model -> no confident match
    bid, sc = M.match_model_id(gemma, "gemini", ["gemma-3-27b-it", "gemini-2.0-flash"])
    assert bid is None

    # nothing gemma-ish at all -> no match
    assert M.match_model_id(gemma, "groq", ["llama-3.3-70b-versatile", "mixtral-8x7b"])[0] is None

    bid, sc = M.match_model_id(qwen, "openrouter", ["qwen/qwen3.8-27b", "qwen/qwen-2.5-72b-instruct"])
    assert bid == "qwen/qwen3.8-27b"

    assert M.looks_same_family(gemma, "google/gemma-4-26b-a4b-it:free") is True
    assert M.looks_same_family(gemma, "gpt-4o") is False
    assert M.looks_same_family(gemma, "gemma-3-27b-it") is False   # different size -> different model
    print("  model-id match: exact alias, size mismatch rejected, family check: OK")


def test_user_defined_profiles():
    import tempfile
    import vlm_config as CFG
    old = CFG.VLM_PROFILES_PATH
    CFG.VLM_PROFILES_PATH = Path(tempfile.mkdtemp()) / "vp.json"
    try:
        assert len(CFG.all_profiles()) == 5 and not CFG.is_user_profile("x")

        CFG.save_user_profiles([{
            "profile_id": "user-g3", "display_name": "My Gemma 3 27B",
            "canonical_model_id": "gemma-3-27b-it",
            "bindings": {
                "gemini": {"model_id": "gemma-3-27b-it", "free_route": True},
                "openrouter": {"model_id": "google/gemma-3-27b-it:free", "free_route": True},
            }}])
        aps = {p.profile_id: p for p in CFG.all_profiles()}
        assert "user-g3" in aps and CFG.is_user_profile("user-g3")
        p = aps["user-g3"]
        assert set(p.bindings) == {"gemini", "openrouter"}
        assert p.bindings["gemini"].identity_status is M.ModelIdentityStatus.UNKNOWN
        assert p.bindings["openrouter"].free_route is True

        # a user profile with the same id as a shipped one overrides it
        CFG.save_user_profiles([
            CFG.load_user_profiles()[0],   # keep user-g3
            {"profile_id": "gemma-4-26b-a4b-it", "display_name": "Gemma (mine)",
             "canonical_model_id": "gemma-3-27b-it",
             "bindings": {"gemini": {"model_id": "gemma-3-27b-it"}}},
        ])
        aps = {p.profile_id: p for p in CFG.all_profiles()}
        assert aps["gemma-4-26b-a4b-it"].display_name == "Gemma (mine)"
        assert len(CFG.all_profiles()) == 6   # 5 shipped (one overridden in place) + user-g3
    finally:
        CFG.VLM_PROFILES_PATH = old
    print("  user-defined profiles: json round-trip, merge, same-id override: OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} VLM PHASE 0/1 TESTS PASSED")
