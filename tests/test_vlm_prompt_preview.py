"""Offline tests for the VLM prompt preview data model."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

# pytest.ini は pytest 実行時のみ src を sys.path へ足す。素の実行でも import できるよう
# ここでも足しておく（末尾の __main__ ランナー用。他テストと同じ方式）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vlm_config import build_generation_profile
from vlm_profiles import GenerationProfile, PromptMode
from vlm_protocols import PROTOCOLS
from vlm_prompt_preview import (
    PromptPreviewRoute,
    build_prompt_preview,
    protocol_placement,
)


def test_default_prompt_text_uses_existing_builders() -> None:
    profile = GenerationProfile()
    preview = build_prompt_preview(profile)
    assert preview.system_prompt
    assert "Write a highly detailed natural-language" in preview.system_prompt
    assert "Write the output in English." in preview.system_prompt
    assert preview.user_prompt == "Caption this image."
    assert not preview.custom_system_prompt_active


def test_custom_prompt_replaces_default_settings() -> None:
    profile = GenerationProfile.from_mapping({
        "custom_system_prompt": "Use only visible colors.",
        "detail_level": "maximum_detail",
    })
    preview = build_prompt_preview(profile)
    assert preview.system_prompt == "Use only visible colors."
    assert preview.custom_system_prompt_active
    prompt_settings = [item for item in preview.settings if item.destination == "prompt"]
    assert prompt_settings
    assert not next(item for item in prompt_settings if item.name == "prompt_mode").overridden
    assert all(item.overridden for item in prompt_settings if item.name != "prompt_mode")


def test_prompt_mode_preview_uses_the_same_builders() -> None:
    preview = build_prompt_preview(GenerationProfile(prompt_mode=PromptMode.SHORT_TAGS))
    assert preview.user_prompt == "Convert this image into concise comma-separated tags."
    assert "comma-separated visual tags" in preview.system_prompt
    settings = {item.name: item for item in preview.settings}
    assert settings["prompt_mode"].value == "short_tags"
    assert settings["detail_level"].ignored_by_mode
    assert not settings["detail_level"].overridden


def test_all_builtin_protocol_placements_are_exposed() -> None:
    preview = build_prompt_preview(GenerationProfile())
    assert [item.protocol for item in preview.placements] == list(PROTOCOLS)
    assert preview.placements[0].system_field == "messages[0].content"
    assert preview.placements[1].max_tokens_field == "max_output_tokens"
    assert preview.placements[2].user_field == "messages[0].content"
    assert preview.placements[3].system_field == "systemInstruction.parts[0].text"


def test_placements_are_derived_from_protocol_classes() -> None:
    for protocol_name, protocol_class in PROTOCOLS.items():
        placement = protocol_placement(protocol_name)
        assert placement is not None
        assert placement.system_field == protocol_class.system_prompt_field
        assert placement.user_field == protocol_class.user_prompt_field
        assert placement.max_tokens_field == protocol_class.max_tokens_field


def test_unknown_route_protocol_shows_openai_chat_fallback_only() -> None:
    preview = build_prompt_preview(
        GenerationProfile(),
        routes=(PromptPreviewRoute("Local", "model-x", "future_protocol"),),
    )
    assert len(preview.placements) == 1
    placement = preview.placements[0]
    assert placement.protocol == "future_protocol"
    assert placement.fallback_protocol == "openai_chat_completions"
    assert placement.system_field == "messages[0].content"


def test_whitespace_padded_unknown_protocol_matches_runtime_fallback() -> None:
    raw_protocol = " future_protocol "
    preview = build_prompt_preview(
        GenerationProfile(),
        routes=(PromptPreviewRoute("Local", "model-x", raw_protocol),),
    )
    placement = preview.placements[0]
    assert placement.protocol == raw_protocol
    assert placement.fallback_protocol == "openai_chat_completions"
    assert placement.system_field == "messages[0].content"


def test_preview_module_is_offline_and_secret_free() -> None:
    source = inspect.getsource(__import__("vlm_prompt_preview"))
    assert "requests" not in source
    assert "httpx" not in source
    assert "api_key" not in source
    assert "base64" not in source.lower()


def test_generation_profile_forwards_optional_settings_when_available() -> None:
    settings = SimpleNamespace(
        generation_profile_id="test",
        language="en",
        detail_level="detailed",
        sentence_mode="automatic_long_detailed",
        character_name_mode="explicit_only",
        markdown="disabled",
        max_output_tokens=2048,
        image_max_long_edge=1024,
        custom_system_prompt="Use concise descriptions.",
        temperature=0.2,
        top_p=0.8,
        image_format="png",
        image_jpeg_quality=75,
    )
    profile = build_generation_profile(settings)
    assert profile.custom_system_prompt == "Use concise descriptions."
    assert profile.temperature == 0.2
    assert profile.top_p == 0.8
    assert profile.image_format == "png"
    assert profile.image_jpeg_quality == 75


def test_generation_profile_keeps_defaults_for_legacy_settings() -> None:
    settings = SimpleNamespace(
        generation_profile_id="legacy",
        language="en",
        detail_level="detailed",
        sentence_mode="automatic_long_detailed",
        character_name_mode="explicit_only",
        markdown="disabled",
        max_output_tokens=2048,
        image_max_long_edge=1024,
    )
    profile = build_generation_profile(settings)
    assert profile.custom_system_prompt == ""
    assert profile.temperature is None
    assert profile.top_p is None
    assert profile.image_format == "auto"
    assert profile.image_jpeg_quality == 90


if __name__ == "__main__":
    tests = [value for name, value in globals().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"VLM prompt preview tests: PASS ({len(tests)} tests)")
