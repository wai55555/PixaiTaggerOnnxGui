"""Read-only VLM prompt preview dialog."""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QLabel,
    QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from vlm_prompt_preview import PromptPreview


class VlmPromptPreviewDialog(QDialog):
    """Show effective prompts and routing metadata without making a request."""

    _SETTING_VALUE_PREFIXES = {
        "prompt_mode": "Opt_PromptMode",
        "language": "Opt_Language",
        "detail_level": "Opt_Detail",
        "sentence_mode": "Opt_Sentence",
        "character_name_mode": "Opt_CharName",
        "markdown": "Opt_Markdown",
        "image_format": "PromptPreview_ImageFormat",
    }

    def __init__(self, preview: PromptPreview, get_string: Callable[..., str],
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.preview = preview
        self._t = get_string
        self.setWindowTitle(self._t("Vlm", "PromptPreview_Title"))
        self.setMinimumSize(420, 360)
        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        if available is None:
            width, height = 760, 680
        else:
            width = max(420, min(760, int(available.width() * 0.90)))
            height = max(400, min(680, int(available.height() * 0.85)))
        self.resize(width, height)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        body = QVBoxLayout(content)

        note = QLabel(self._t("Vlm", "PromptPreview_Note"))
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        body.addWidget(note)

        summary = QGroupBox(self._t("Vlm", "PromptPreview_Summary"))
        summary_form = QFormLayout(summary)
        summary_form.addRow(
            self._t("Vlm", "PromptPreview_Profile"),
            QLabel(self.preview.profile_id or self._t("Vlm", "PromptPreview_Default")),
        )
        body.addWidget(summary)

        system_group = QGroupBox(self._t("Vlm", "PromptPreview_SystemPrompt"))
        system_layout = QVBoxLayout(system_group)
        self.system_edit = self._prompt_edit(self.preview.system_prompt)
        self.system_prompt_edit = self.system_edit
        system_layout.addWidget(self.system_edit)
        if self.preview.custom_system_prompt_active:
            custom_note = QLabel(self._t("Vlm", "PromptPreview_CustomOverride"))
            custom_note.setWordWrap(True)
            custom_note.setStyleSheet("color: gray;")
            system_layout.addWidget(custom_note)
        system_layout.addWidget(self._copy_button(
            self._t("Vlm", "PromptPreview_CopySystem"), self.preview.system_prompt))
        body.addWidget(system_group)

        user_group = QGroupBox(self._t("Vlm", "PromptPreview_UserPrompt"))
        user_layout = QVBoxLayout(user_group)
        self.user_edit = self._prompt_edit(self.preview.user_prompt)
        self.user_prompt_edit = self.user_edit
        user_layout.addWidget(self.user_edit)
        user_layout.addWidget(self._copy_button(
            self._t("Vlm", "PromptPreview_CopyUser"), self.preview.user_prompt))
        body.addWidget(user_group)

        settings_group = QGroupBox(self._t("Vlm", "PromptPreview_Destinations"))
        settings_form = QFormLayout(settings_group)
        for item in self.preview.settings:
            name = self._t("Vlm", f"PromptPreview_Setting_{item.name}")
            value = self._localized_setting_value(item.name, item.value)
            destination = self._t("Vlm", f"PromptPreview_Destination_{item.destination}")
            if item.overridden:
                value = f"{value} ({self._t('Vlm', 'PromptPreview_Overridden')})"
            elif item.ignored_by_mode:
                value = f"{value} ({self._t('Vlm', 'PromptPreview_IgnoredByMode')})"
            settings_form.addRow(name, QLabel(f"{value}  [{destination}]"))
        body.addWidget(settings_group)

        route_group = QGroupBox(self._t("Vlm", "PromptPreview_Routes"))
        route_layout = QVBoxLayout(route_group)
        route_note = QLabel(self._t("Vlm", "PromptPreview_Routes_Note"))
        route_note.setWordWrap(True)
        route_note.setStyleSheet("color: gray;")
        route_layout.addWidget(route_note)
        if self.preview.routes:
            for route in self.preview.routes:
                label = QLabel(self._t(
                    "Vlm", "PromptPreview_Route",
                    connection=route.connection_name or self._t("Vlm", "PromptPreview_Unknown"),
                    model=route.model_id or self._t("Vlm", "PromptPreview_Unset"),
                    protocol=route.protocol or self._t("Vlm", "PromptPreview_Unknown"),
                ))
                label.setWordWrap(True)
                route_layout.addWidget(label)
        else:
            route_layout.addWidget(QLabel(self._t("Vlm", "PromptPreview_NoRoutes")))
        for placement in self.preview.placements:
            fallback = ""
            if placement.fallback_protocol:
                fallback = self._t(
                    "Vlm", "PromptPreview_Fallback",
                    protocol=placement.fallback_protocol,
                )
            label = QLabel(self._t(
                "Vlm", "PromptPreview_Placement",
                protocol=placement.protocol,
                system=placement.system_field,
                user=placement.user_field,
                max_tokens=placement.max_tokens_field,
                fallback=fallback,
            ))
            label.setWordWrap(True)
            route_layout.addWidget(label)
        body.addWidget(route_group)

        copy_all = QPushButton(self._t("Vlm", "PromptPreview_CopyAll"))
        copy_all.clicked.connect(lambda: QGuiApplication.clipboard().setText(
            f"{self._t('Vlm', 'PromptPreview_SystemPrompt')}:\n{self.preview.system_prompt}\n\n"
            f"{self._t('Vlm', 'PromptPreview_UserPrompt')}:\n{self.preview.user_prompt}"))
        body.addWidget(copy_all)
        body.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _localized_setting_value(self, name: str, value: str | None) -> str:
        if value is None:
            return self._t("Vlm", "PromptPreview_ProviderDefault")
        prefix = self._SETTING_VALUE_PREFIXES.get(name)
        if prefix is None:
            return value
        key = f"{prefix}_{value}"
        try:
            localized = self._t("Vlm", key)
        except (KeyError, TypeError, ValueError):
            return value
        # LocaleManager returns the key itself when a section/key is unavailable.
        return value if not localized or localized == key else localized

    @staticmethod
    def _prompt_edit(text: str) -> QPlainTextEdit:
        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(text)
        edit.setMinimumHeight(100)
        edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        return edit

    @staticmethod
    def _copy_button(label: str, text: str) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(lambda: QGuiApplication.clipboard().setText(text))
        return button
