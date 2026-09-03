"""モデルプロファイルの新規作成／編集（260901_VLM_requirement FR-003）。

出荷プロファイルは推定なので、利用者が「モデル一覧」で見つけた実 ID を各プロバイダーに
割り当てて自分のプロファイルを作れるようにする。フォールバックは「同一モデルを複数
プロバイダーで回す」設計なので、canonical モデル ID を軸に、各経路がそのモデルの
どの ID かを対応づける画面。
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

import vlm_config
import vlm_models
import vlm_secrets
from vlm_connections import default_builtin_connections
from vlm_worker import VlmModelListWorker

GetString = Callable[..., str]
_PROVIDERS = (
    "gemini", "openrouter", "cloudflare", "groq", "nvidia", "mistral",
    "huggingface", "vercel", "openai", "anthropic",
    # "ovhcloud",  # 日本居住者環境で実機検証できるまで無効
)
_SECRET_REF = {
    "gemini": "vlm/gemini/api_key", "openrouter": "vlm/openrouter/api_key",
    "cloudflare": "vlm/cloudflare/api_token", "groq": "vlm/groq/api_key",
    "nvidia": "vlm/nvidia/api_key", "mistral": "vlm/mistral/api_key",
    "huggingface": "vlm/huggingface/api_token",
    "vercel": "vlm/vercel/api_key", "openai": "vlm/openai/api_key",
    "anthropic": "vlm/anthropic/api_key",
    # "ovhcloud": "vlm/ovhcloud/api_key",
}


class ProfileEditorDialog(QDialog):
    def __init__(self, get_string: GetString, profile_dict: dict | None,
                 *, read_only: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self._t = get_string
        self._src = dict(profile_dict or {})
        self._read_only = read_only
        self._result: dict | None = None
        self._ml_thread: QThread | None = None
        self._ml_worker: VlmModelListWorker | None = None
        self._ml_provider = ""
        self.setWindowTitle(get_string("Vlm", "ProfileEdit_Title"))
        self.setMinimumWidth(560)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(self._src.get("display_name", ""))
        self.canon_edit = QLineEdit(self._src.get("canonical_model_id", ""))
        self.canon_edit.setPlaceholderText(self._t("Vlm", "ProfileEdit_Canonical_Placeholder"))
        form.addRow(self._t("Vlm", "ProfileEdit_Name"), self.name_edit)
        form.addRow(self._t("Vlm", "ProfileEdit_Canonical"), self.canon_edit)
        root.addLayout(form)

        root.addWidget(QLabel(self._t("Vlm", "ProfileEdit_Routes_Hint")))
        src_bindings = self._src.get("bindings") or {}
        self._rows: dict[str, dict] = {}
        for prov in _PROVIDERS:
            b = src_bindings.get(prov) or {}
            row = QHBoxLayout()
            inc = QCheckBox(prov)
            inc.setChecked(bool(b))
            inc.setFixedWidth(96)
            mid = QComboBox()
            mid.setEditable(True)
            mid.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            mid.setMinimumWidth(230)
            if b.get("model_id"):
                mid.addItem(str(b["model_id"]))
                mid.setCurrentText(str(b["model_id"]))
            fetch = QPushButton(self._t("Vlm", "Settings_Route_FetchModels"))
            fetch.clicked.connect(lambda _=False, p=prov: self._fetch(p))
            free = QCheckBox(self._t("Vlm", "ProfileEdit_Free"))
            free.setChecked(bool(b.get("free_route", False)))
            status = QLabel()
            for w in (inc, mid, fetch, free):
                row.addWidget(w)
            row.addWidget(status, 1)
            root.addLayout(row)
            self._rows[prov] = {"inc": inc, "mid": mid, "fetch": fetch, "free": free, "status": status}

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self._on_save)
        box.rejected.connect(self.reject)
        root.addWidget(box)
        if self._read_only:
            for r in self._rows.values():
                for k in ("inc", "mid", "fetch", "free"):
                    r[k].setEnabled(False)
            self.name_edit.setReadOnly(True)
            self.canon_edit.setReadOnly(True)
            box.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)

    # --- model list ----------------------------------------------------------
    def _fetch(self, provider: str) -> None:
        if self._ml_thread is not None:
            return
        conn = next((c for c in default_builtin_connections() if c.provider_id == provider), None)
        if conn is None:
            return
        mid = self._rows[provider]["mid"].currentText().strip()
        if mid:
            conn.model_id = mid
        if "{account_id}" in conn.base_url:
            self._rows[provider]["status"].setText(self._t("Vlm", "ProfileEdit_Cf_Needs_Account"))
            return
        api_key = vlm_secrets.get_secret(_SECRET_REF.get(provider, "")) or None
        self._rows[provider]["status"].setText(self._t("Vlm", "Settings_Route_FetchModels_Busy"))
        self._ml_provider = provider
        self._ml_thread = QThread(self)
        self._ml_worker = VlmModelListWorker(conn, api_key)
        self._ml_worker.moveToThread(self._ml_thread)
        self._ml_thread.started.connect(self._ml_worker.run)
        self._ml_worker.result_ready.connect(self._on_models)
        self._ml_worker.finished.connect(self._ml_thread.quit)
        self._ml_thread.finished.connect(self._ml_done)
        for r in self._rows.values():
            r["fetch"].setEnabled(False)
        self._ml_thread.start()

    @Slot(str, object)
    def _on_models(self, _cid: str, result) -> None:
        prov = self._ml_provider
        r = self._rows.get(prov)
        if r is None:
            return
        if not isinstance(result, list):
            r["status"].setText(self._t("Vlm", "Settings_Route_FetchModels_Fail",
                                        detail=getattr(result, "message", "") or str(result)))
            return
        combo = r["mid"]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(result)
        combo.blockSignals(False)
        canon = self.canon_edit.text().strip()
        probe = vlm_models.VlmModelProfile(
            profile_id="_probe", display_name="_", canonical_model_id=canon or "x",
            base_model=canon, aliases=(canon,) if canon else ())
        best, score = vlm_models.match_model_id(probe, prov, result)
        okmsg = self._t("Vlm", "Settings_Route_FetchModels_Ok", n=len(result))
        if best is not None:
            combo.setCurrentText(best)
            r["inc"].setChecked(True)
            r["status"].setText(okmsg + " — " + self._t(
                "Vlm", "Settings_Route_FetchModels_Exact" if score >= 0.999
                else "Settings_Route_FetchModels_Matched", id=best))
        else:
            r["status"].setText(okmsg + " — " + self._t(
                "Vlm", "Settings_Route_FetchModels_NoMatch",
                profile=canon or self.name_edit.text() or "?"))

    def _ml_done(self) -> None:
        if self._ml_worker is not None:
            self._ml_worker.deleteLater()
            self._ml_worker = None
        if self._ml_thread is not None:
            self._ml_thread.deleteLater()
            self._ml_thread = None
        for r in self._rows.values():
            r["fetch"].setEnabled(not self._read_only)

    def _await_ml(self) -> None:
        th = self._ml_thread
        if th is not None and th.isRunning():
            try:
                self._ml_worker.result_ready.disconnect()
            except (RuntimeError, TypeError):
                pass
            th.quit()
            th.wait(30000)

    def done(self, r: int) -> None:
        self._await_ml()
        super().done(r)

    # --- save --------------------------------------------------------------
    def _on_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, self.windowTitle(), self._t("Vlm", "ProfileEdit_Need_Name"))
            return
        bindings: dict[str, dict] = {}
        for prov, r in self._rows.items():
            if not r["inc"].isChecked():
                continue
            mid = r["mid"].currentText().strip()
            if not mid:
                continue
            bindings[prov] = {"model_id": mid, "free_route": r["free"].isChecked()}
        if not bindings:
            QMessageBox.warning(self, self.windowTitle(), self._t("Vlm", "ProfileEdit_Need_Route"))
            return
        pid = self._src.get("profile_id") or vlm_config.new_profile_id(name)
        self._result = {
            "profile_id": pid,
            "display_name": name,
            "canonical_model_id": self.canon_edit.text().strip() or name,
            "bindings": bindings,
        }
        self.accept()

    def result_profile(self) -> dict | None:
        return self._result
