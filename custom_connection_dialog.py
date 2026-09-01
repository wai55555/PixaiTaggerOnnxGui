"""カスタム接続編集ダイアログ（260901_VLM_design.md 6.4節 / implement_plan 8章）。

外部・ローカルを含む任意の VLM 接続を1件編集する。秘密値はこのダイアログでは
vlm_secrets 経由でのみ扱い、返す dict には含めない。
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QMessageBox, QSpinBox, QVBoxLayout, QWidget,
)

import vlm_secrets
from vlm_connections import ConnectionLocality, resolve_custom_kind
from vlm_config import new_connection_id

GetString = Callable[..., str]

_PROTOCOLS = [
    ("openai_chat_completions", "OpenAI Chat Completions"),
    ("gemini_generate_content", "Google Gemini generateContent"),
]
_AUTH_TYPES = [
    ("none", "None"),
    ("bearer", "Bearer token"),
    ("header_key", "API key in header"),
    ("query_key", "API key in query"),
]
_LOCALITY = [
    (ConnectionLocality.AUTO, "Auto"),
    (ConnectionLocality.LOCAL, "Local"),
    (ConnectionLocality.EXTERNAL, "External"),
]


class CustomConnectionDialog(QDialog):
    def __init__(self, get_string: GetString, existing: dict | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._t = get_string
        self._existing = dict(existing or {})
        self._result: dict | None = None
        self.setWindowTitle(get_string("Vlm", "Custom_Dialog_Title"))
        self.setMinimumWidth(460)
        self._build()
        self._load(self._existing)

    # --- UI ---
    def _build(self) -> None:
        root = QVBoxLayout(self)

        basic = QGroupBox(self._t("Vlm", "Custom_Section_Basic"))
        bf = QFormLayout(basic)
        self.name_edit = QLineEdit()
        self.locality_combo = _combo(_LOCALITY)
        self.protocol_combo = _combo(_PROTOCOLS)
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("http://127.0.0.1:1234/v1")
        self.model_edit = QLineEdit()
        bf.addRow(self._t("Vlm", "Custom_Field_Name"), self.name_edit)
        bf.addRow(self._t("Vlm", "Custom_Field_Locality"), self.locality_combo)
        bf.addRow(self._t("Vlm", "Custom_Field_Protocol"), self.protocol_combo)
        bf.addRow(self._t("Vlm", "Custom_Field_BaseUrl"), self.base_url_edit)
        bf.addRow(self._t("Vlm", "Custom_Field_ModelId"), self.model_edit)
        root.addWidget(basic)

        auth = QGroupBox(self._t("Vlm", "Custom_Section_Auth"))
        af = QFormLayout(auth)
        self.auth_type_combo = _combo(_AUTH_TYPES)
        self.auth_header_edit = QLineEdit("Authorization")
        self.auth_query_edit = QLineEdit("key")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText(self._t("Vlm", "Custom_ApiKey_Placeholder"))
        self.persist_key_check = QCheckBox(self._t("Vlm", "Custom_ApiKey_Persist"))
        self.persist_key_check.setChecked(vlm_secrets.keyring_available())
        self.verify_tls_check = QCheckBox(self._t("Vlm", "Custom_Field_VerifyTls"))
        self.verify_tls_check.setChecked(True)
        af.addRow(self._t("Vlm", "Custom_Field_AuthType"), self.auth_type_combo)
        af.addRow(self._t("Vlm", "Custom_Field_AuthHeader"), self.auth_header_edit)
        af.addRow(self._t("Vlm", "Custom_Field_AuthQuery"), self.auth_query_edit)
        af.addRow(self._t("Vlm", "Custom_Field_ApiKey"), self.api_key_edit)
        af.addRow("", self.persist_key_check)
        af.addRow("", self.verify_tls_check)
        root.addWidget(auth)

        adv = QGroupBox(self._t("Vlm", "Custom_Section_Advanced"))
        gf = QFormLayout(adv)
        self.connect_timeout = _spin(1, 120, 10)
        self.read_timeout = _spin(5, 900, 120)
        self.retry_same = _spin(0, 5, 1)
        self.concurrency = _spin(1, 16, 1)
        self.text_path_edit = QLineEdit()
        self.text_path_edit.setPlaceholderText("choices[0].message.content")
        self.max_edge = _spin(256, 8192, 1536)
        gf.addRow(self._t("Vlm", "Custom_Field_ConnectTimeout"), self.connect_timeout)
        gf.addRow(self._t("Vlm", "Custom_Field_ReadTimeout"), self.read_timeout)
        gf.addRow(self._t("Vlm", "Custom_Field_RetrySame"), self.retry_same)
        gf.addRow(self._t("Vlm", "Custom_Field_Concurrency"), self.concurrency)
        gf.addRow(self._t("Vlm", "Custom_Field_TextPath"), self.text_path_edit)
        gf.addRow(self._t("Vlm", "Custom_Field_MaxEdge"), self.max_edge)
        root.addWidget(adv)

        root.addWidget(QLabel(self._t("Vlm", "Custom_Header_Note")))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.auth_type_combo.currentIndexChanged.connect(self._sync_auth_rows)
        self._sync_auth_rows()

    def _sync_auth_rows(self) -> None:
        atype = self.auth_type_combo.currentData()
        self.auth_header_edit.setEnabled(atype in ("bearer", "header_key"))
        self.auth_query_edit.setEnabled(atype == "query_key")
        self.api_key_edit.setEnabled(atype != "none")
        self.persist_key_check.setEnabled(atype != "none" and vlm_secrets.keyring_available())

    # --- data ---
    def _load(self, data: dict) -> None:
        self.name_edit.setText(str(data.get("display_name", "")))
        _select(self.protocol_combo, data.get("protocol", "openai_chat_completions"))
        self.base_url_edit.setText(str(data.get("base_url", "")))
        self.model_edit.setText(str(data.get("model_id", "")))
        auth = data.get("auth", {}) if isinstance(data.get("auth"), dict) else {}
        _select(self.auth_type_combo, auth.get("type", "none"))
        self.auth_header_edit.setText(str(auth.get("header_name", "Authorization")))
        self.auth_query_edit.setText(str(auth.get("query_param", "key")))
        retry = data.get("retry", {}) if isinstance(data.get("retry"), dict) else {}
        self.connect_timeout.setValue(_int(retry.get("connect_timeout_s"), 10))
        self.read_timeout.setValue(_int(retry.get("read_timeout_s"), 120))
        self.retry_same.setValue(_int(retry.get("retry_same_max"), 1))
        self.concurrency.setValue(_int(data.get("concurrency"), 1))
        resp = data.get("response", {}) if isinstance(data.get("response"), dict) else {}
        self.text_path_edit.setText(str(resp.get("text_path", "") or data.get("text_path", "")))
        self.verify_tls_check.setChecked(bool(data.get("verify_tls", True)))
        img = data.get("image", {}) if isinstance(data.get("image"), dict) else {}
        self.max_edge.setValue(_int(img.get("max_long_edge"), 1536))
        kind_raw = str(data.get("kind", "")).lower()
        if kind_raw == "custom_local":
            _select_enum(self.locality_combo, ConnectionLocality.LOCAL)
        elif kind_raw == "custom_external":
            _select_enum(self.locality_combo, ConnectionLocality.EXTERNAL)
        else:
            _select_enum(self.locality_combo, ConnectionLocality.AUTO)
        self._sync_auth_rows()

    def _on_save(self) -> None:
        name = self.name_edit.text().strip()
        base_url = self.base_url_edit.text().strip()
        model_id = self.model_edit.text().strip()
        if not name or not base_url or not model_id:
            QMessageBox.warning(self, self._t("Vlm", "Custom_Dialog_Title"),
                                self._t("Vlm", "Custom_Validation_Required"))
            return

        cid = self._existing.get("connection_id") or new_connection_id()
        locality = self.locality_combo.currentData()
        kind = resolve_custom_kind(locality, base_url)
        atype = self.auth_type_combo.currentData()
        secret_ref = self._existing.get("auth", {}).get("secret_ref") if isinstance(self._existing.get("auth"), dict) else ""
        secret_ref = secret_ref or (f"vlm/custom/{cid}" if atype != "none" else "")

        if atype != "none":
            key = self.api_key_edit.text().strip()
            if key:
                vlm_secrets.set_secret(secret_ref, key, persist=self.persist_key_check.isChecked())
        else:
            # 認証なしに変更したら、以前保存した鍵は残さない。
            old_ref = self._existing.get("auth", {}).get("secret_ref") if isinstance(self._existing.get("auth"), dict) else ""
            if old_ref:
                vlm_secrets.delete_secret(old_ref)

        self._result = {
            "connection_id": cid,
            "display_name": name,
            "kind": kind.value,
            "protocol": self.protocol_combo.currentData(),
            "base_url": base_url,
            "model_id": model_id,
            "enabled": bool(self._existing.get("enabled", True)),
            "verify_tls": self.verify_tls_check.isChecked(),
            "concurrency": self.concurrency.value(),
            "auth": {
                "type": atype,
                "secret_ref": secret_ref,
                "header_name": self.auth_header_edit.text().strip() or "Authorization",
                "query_param": self.auth_query_edit.text().strip() or "key",
            },
            "retry": {
                "connect_timeout_s": self.connect_timeout.value(),
                "read_timeout_s": self.read_timeout.value(),
                "retry_same_max": self.retry_same.value(),
            },
            "response": {"text_path": self.text_path_edit.text().strip()},
            "image": {"max_long_edge": self.max_edge.value()},
        }
        self.accept()

    def result_connection(self) -> dict | None:
        return self._result


def _combo(pairs) -> QComboBox:
    c = QComboBox()
    for value, label in pairs:
        c.addItem(label, value)
    return c


def _select(combo: QComboBox, value) -> None:
    idx = combo.findData(value)
    combo.setCurrentIndex(idx if idx >= 0 else 0)


def _select_enum(combo: QComboBox, value) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
    combo.setCurrentIndex(0)


def _int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _spin(lo: int, hi: int, val: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    return s
