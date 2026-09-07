"""VLM prompt preview data (offline, secret-free).

The preview deliberately contains only the text prompts and the small amount of
metadata needed to explain where each prompt is placed.  Request construction,
authentication, image encoding, and network access do not belong here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vlm_profiles import GenerationProfile, PromptMode, build_system_prompt, build_user_prompt
from vlm_protocols import PROTOCOLS


@dataclass(frozen=True)
class PromptPreviewSetting:
    """One effective setting and the part of the request it affects."""

    name: str
    value: str | None
    destination: str  # prompt | api_parameter | image_preprocess
    overridden: bool = False
    ignored_by_mode: bool = False


@dataclass(frozen=True)
class PromptPreviewRoute:
    """Non-sensitive route information shown in the provider placement section."""

    connection_name: str
    model_id: str
    protocol: str


@dataclass(frozen=True)
class ProtocolPlacement:
    """Where prompt text is placed for one existing protocol."""

    protocol: str
    system_field: str
    user_field: str
    max_tokens_field: str
    fallback_protocol: str = ""


@dataclass(frozen=True)
class PromptPreview:
    """Complete read-only data used by the preview dialog."""

    profile_id: str
    system_prompt: str
    user_prompt: str
    settings: tuple[PromptPreviewSetting, ...]
    routes: tuple[PromptPreviewRoute, ...]
    placements: tuple[ProtocolPlacement, ...]
    custom_system_prompt_active: bool = False


def protocol_placement(protocol: str) -> ProtocolPlacement | None:
    """Return placement metadata for an existing protocol, if supported."""
    protocol_name = str(protocol or "")
    protocol_class = PROTOCOLS.get(protocol_name)
    if protocol_class is None:
        return None
    return ProtocolPlacement(
        protocol=protocol_name,
        system_field=protocol_class.system_prompt_field,
        user_field=protocol_class.user_prompt_field,
        max_tokens_field=protocol_class.max_tokens_field,
    )


def _preview_placement(protocol: str) -> ProtocolPlacement:
    """Describe the placement used by execution, including unknown-name fallback."""
    # Keep the raw name: vlm_protocols.get_protocol() does not normalize whitespace
    # before its dictionary lookup, so the preview must describe that exact fallback.
    raw = str(protocol or "")
    known = protocol_placement(raw)
    if known is not None:
        return known
    fallback = protocol_placement("openai_chat_completions")
    if fallback is None:  # PROTOCOLS always contains the runtime fallback.
        raise RuntimeError("openai_chat_completions protocol is not registered")
    return ProtocolPlacement(
        protocol=raw or "(unknown)",
        system_field=fallback.system_field,
        user_field=fallback.user_field,
        max_tokens_field=fallback.max_tokens_field,
        fallback_protocol=fallback.protocol,
    )


def build_prompt_preview(
    profile: GenerationProfile,
    *,
    routes: Iterable[PromptPreviewRoute] = (),
    protocols: Iterable[str] | None = None,
) -> PromptPreview:
    """Build preview data from the same prompt builders used by VLM execution."""
    route_tuple = tuple(routes)
    protocol_names = (
        tuple(protocols)
        if protocols is not None
        else tuple(PROTOCOLS)
    )
    if protocols is None:
        if route_tuple:
            placement_names = dict.fromkeys(route.protocol for route in route_tuple)
        else:
            placement_names = dict.fromkeys(PROTOCOLS)
    else:
        placement_names = dict.fromkeys(protocol_names)
    placements = tuple(_preview_placement(protocol) for protocol in placement_names)
    custom_system_prompt_active = bool(profile.custom_system_prompt.strip())
    settings = (
        PromptPreviewSetting("prompt_mode", profile.prompt_mode.value, "prompt",
                             custom_system_prompt_active),
        PromptPreviewSetting("language", profile.language, "prompt", custom_system_prompt_active),
        PromptPreviewSetting("detail_level", profile.detail_level.value, "prompt",
                             custom_system_prompt_active,
                             profile.prompt_mode is not PromptMode.STANDARD),
        PromptPreviewSetting("sentence_mode", profile.sentence_mode.value, "prompt",
                             custom_system_prompt_active,
                             profile.prompt_mode is not PromptMode.STANDARD),
        PromptPreviewSetting("character_name_mode", profile.character_name_mode.value, "prompt",
                             custom_system_prompt_active,
                             profile.prompt_mode is not PromptMode.STANDARD),
        PromptPreviewSetting("markdown", profile.markdown.value, "prompt",
                             custom_system_prompt_active,
                             profile.prompt_mode is not PromptMode.STANDARD),
        PromptPreviewSetting("max_output_tokens", str(profile.max_output_tokens), "api_parameter"),
        PromptPreviewSetting("temperature", _display_optional(profile.temperature), "api_parameter"),
        PromptPreviewSetting("top_p", _display_optional(profile.top_p), "api_parameter"),
        PromptPreviewSetting("image_max_long_edge", str(profile.image_max_long_edge), "image_preprocess"),
        PromptPreviewSetting("image_format", profile.image_format, "image_preprocess"),
        PromptPreviewSetting("image_jpeg_quality", str(profile.image_jpeg_quality), "image_preprocess"),
    )
    return PromptPreview(
        profile_id=profile.profile_id,
        system_prompt=build_system_prompt(profile),
        user_prompt=build_user_prompt(profile),
        settings=settings,
        routes=route_tuple,
        placements=placements,
        custom_system_prompt_active=custom_system_prompt_active,
    )


def _display_optional(value: float | None) -> str | None:
    """Return None for provider defaults; the dialog supplies the localized label."""
    return None if value is None else str(value)
