"""APIキー登録ダイアログ（260901_VLM_spec.md 15章 / design.md 6.3節）。

「取得ページを開く → 貼り付け → 自動チェック → 保存して閉じる」を1画面で完結させる。
未ログイン/未サインアップのユーザーも迷わないよう、手順を文章で示す。
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

import vlm_secrets
from vlm_diagnostics import DiagStatus
from vlm_worker import VlmDiagnosticsWorker

GetString = Callable[..., str]


class ApiKeyDialog(QDialog):
    def __init__(self, get_string: GetString, *, display_name: str, secret_ref: str,
                 conn, key_url: str, login_url: str, instructions: str,
                 cloudflare_account_id: str = "",
                 on_cloudflare_verified: Callable[[str], None] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._t = get_string
        self._secret_ref = secret_ref
        self._conn = conn
        self._key_url = key_url
        self._login_url = login_url
        self._is_cloudflare = getattr(conn, "provider_id", "") == "cloudflare"
        self._cloudflare_account_id = cloudflare_account_id
        self._on_cloudflare_verified = on_cloudflare_verified
        self._saved = False
        self._check_thread: QThread | None = None
        self._check_worker: VlmDiagnosticsWorker | None = None
        self.setWindowTitle(get_string("Vlm", "ApiKey_Title", service=display_name))
        self.setMinimumWidth(460)
        self._build(display_name, instructions)

    def _build(self, display_name: str, instructions: str) -> None:
        root = QVBoxLayout(self)

        head = QLabel(self._t("Vlm", "ApiKey_Header", service=display_name))
        head.setWordWrap(True)
        root.addWidget(head)

        # locale の値に含まれる "\n" リテラルを実改行へ（configparser はエスケープを展開しない）。
        steps = QLabel(instructions.replace("\\n", "\n"))
        steps.setWordWrap(True)
        steps.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(steps)

        link_row = QHBoxLayout()
        open_key = QPushButton(self._t("Vlm", "ApiKey_Open_Key_Page"))
        open_key.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._key_url)))
        link_row.addWidget(open_key)
        if self._login_url and self._login_url != self._key_url:
            open_login = QPushButton(self._t("Vlm", "ApiKey_Open_Login_Page"))
            open_login.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._login_url)))
            link_row.addWidget(open_login)
        link_row.addStretch(1)
        root.addLayout(link_row)

        url_hint = QLabel(f"{self._key_url}")
        url_hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        url_hint.setStyleSheet("color: gray;")
        root.addWidget(url_hint)

        # 既に登録済みなら、それを（値は伏せたまま）明示する。空欄だと未登録に見える。
        where = vlm_secrets.secret_status(self._secret_ref)
        registered = where != "missing"
        where_label = {
            "keyring": self._t("Vlm", "ApiKey_Where_Keyring"),
            "env": self._t("Vlm", "ApiKey_Where_Env"),
            "session": self._t("Vlm", "ApiKey_Where_Session"),
        }.get(where, where)
        registered_key = "ApiKey_Current_Registered_Cloudflare" if self._is_cloudflare \
            else "ApiKey_Current_Registered"
        self.current_label = QLabel(
            self._t("Vlm", registered_key, where=where_label) if registered
            else self._t("Vlm", "ApiKey_Current_None"))
        self.current_label.setStyleSheet("color: gray;")
        root.addWidget(self.current_label)

        self.account_id_edit: QLineEdit | None = None
        if self._is_cloudflare:
            account_row = QHBoxLayout()
            account_row.addWidget(QLabel(self._t("Vlm", "Settings_Cf_Account")))
            self.account_id_edit = QLineEdit(self._cloudflare_account_id)
            self.account_id_edit.setPlaceholderText(self._t("Vlm", "ApiKey_Cf_Account_Placeholder"))
            account_row.addWidget(self.account_id_edit, 1)
            root.addLayout(account_row)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText(self._t(
            "Vlm", "ApiKey_Paste_Placeholder_Update" if registered else "ApiKey_Paste_Placeholder"))
        self.key_edit.returnPressed.connect(self._check_and_save)
        root.addWidget(self.key_edit)

        self.save_btn = QPushButton(self._t("Vlm", "ApiKey_Verify_And_Save"))
        self.save_btn.clicked.connect(self._check_and_save)
        root.addWidget(self.save_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        box.rejected.connect(self.reject)
        root.addWidget(box)

    # --- verify -------------------------------------------------------------
    def _check_and_save(self) -> None:
        if self._check_thread is not None:
            return
        entered_key = self.key_edit.text().strip()
        key = entered_key
        if not key and self._is_cloudflare:
            key = vlm_secrets.get_secret(self._secret_ref) or ""
        if not key:
            self.status.setText(self._t("Vlm", "ApiKey_Empty"))
            return
        check_conn = self._conn
        if self._is_cloudflare:
            account_id = self.account_id_edit.text().strip() if self.account_id_edit else ""
            if not account_id:
                self.status.setText(self._t("Vlm", "ApiKey_Cf_Account_Empty"))
                return
            if re.fullmatch(r"[0-9a-fA-F]{32}", account_id) is None:
                self.status.setText(self._t("Vlm", "ApiKey_Cf_Account_Invalid"))
                return
            self._pending_account_id = account_id
            check_conn = replace(
                self._conn,
                base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
            )
        self._set_busy(True)
        self.status.setText(self._t("Vlm", "ApiKey_Checking"))
        self._pending_key = key
        self._pending_key_is_new = bool(entered_key)
        self._check_thread = QThread(self)
        self._check_worker = VlmDiagnosticsWorker(check_conn, key)
        self._check_worker.moveToThread(self._check_thread)
        self._check_thread.started.connect(self._check_worker.run)
        self._check_worker.report_ready.connect(self._on_report)
        self._check_worker.finished.connect(self._check_thread.quit)
        self._check_thread.finished.connect(self._on_check_thread_done)
        self._check_thread.start()

    def _on_report(self, report) -> None:
        """キーが使えるかの判定。サーバーが 401/403 で弾いた＝キー不正。到達不能／
        DNS・TLS 失敗＝未成立で保存しない。それ以外はサーバーが応答している＝キーは
        認証を通っているので保存する（モデル ID 違い等は警告どまりで別途直す）。"""
        self._verify_failed = ""
        self._model_warning = ""
        items = {i.name: i for i in report.items}
        auth = items.get("Auth")
        http = items.get("HTTP response")
        extraction = items.get("Caption extraction")

        if self._is_cloudflare:
            # Account ID・トークン・Workers AI 権限・モデル・画像入力・応答抽出の全部が
            # 通ったときだけ保存する。キー単体の token/verify 成功では閉じない。
            if (http is not None and http.status is DiagStatus.PASS
                    and extraction is not None and extraction.status is DiagStatus.PASS):
                return
            self._verify_failed = (
                (http.detail if http is not None and http.status is not DiagStatus.PASS else "")
                or (extraction.detail if extraction is not None else "")
                or (auth.detail if auth is not None and auth.status is DiagStatus.FAIL else "")
                or self._t("Vlm", "ApiKey_Failed_Generic")
            )
            return

        if http is not None and http.status is DiagStatus.PASS:
            return   # 200: キーもモデルも通った
        if (auth is not None and auth.status is DiagStatus.FAIL) or report.http_status in (401, 403):
            self._verify_failed = ((http.detail if http is not None else None)
                                   or (auth.detail if auth is not None else None)
                                   or self._t("Vlm", "ApiKey_Failed_Generic"))
            return
        if report.http_status is not None:
            # サーバーが 401/403 以外で応答した＝キーは認証を通っている。
            # 200 でない理由（モデル ID 違い等）は警告どまりで、キーは保存する。
            self._model_warning = ((http.detail if http is not None else None)
                                   or f"HTTP {report.http_status}")
            return
        # 実応答まで到達しなかった（DNS/TLS 失敗・到達不能）→ 保存しない
        for name in ("TLS", "DNS / TCP", "Request build"):
            it = items.get(name)
            if it is not None and it.status is DiagStatus.FAIL:
                self._verify_failed = it.detail or self._t("Vlm", "ApiKey_Failed_Generic")
                return
        self._verify_failed = self._t("Vlm", "ApiKey_Not_Reached")

    def _on_check_thread_done(self) -> None:
        if self._check_worker is not None:
            self._check_worker.deleteLater()
            self._check_worker = None
        if self._check_thread is not None:
            self._check_thread.deleteLater()
            self._check_thread = None
        failed = getattr(self, "_verify_failed", self._t("Vlm", "ApiKey_Failed_Generic"))
        if failed:
            self._set_busy(False)
            self.status.setText(self._t("Vlm", "ApiKey_Failed", detail=failed))
            return
        # 成功: 保存して確認メッセージ → 閉じる。persisted は「keyring へ永続保存できたか」。
        persisted = False
        if getattr(self, "_pending_key_is_new", True):
            persisted = vlm_secrets.set_secret(self._secret_ref, self._pending_key,
                                               persist=vlm_secrets.keyring_available())
            where = self._t("Vlm", "ApiKey_Stored_Keyring" if persisted else "ApiKey_Stored_Session")
            success_msg = self._t("Vlm", "ApiKey_Saved_Confirm", where=where)
        else:
            success_msg = self._t("Vlm", "ApiKey_Connected_Existing")
        if self._is_cloudflare and self._on_cloudflare_verified is not None:
            self._on_cloudflare_verified(self._pending_account_id)
        self._saved = True
        warn = getattr(self, "_model_warning", "")
        if warn:
            where = self._t("Vlm", "ApiKey_Stored_Keyring" if persisted else "ApiKey_Stored_Session")
            msg = self._t("Vlm", "ApiKey_Saved_Model_Warning", where=where, detail=warn)
        else:
            msg = success_msg
        QMessageBox.information(self, self.windowTitle(), msg)
        self.accept()

    def _set_busy(self, busy: bool) -> None:
        self.key_edit.setEnabled(not busy)
        if self.account_id_edit is not None:
            self.account_id_edit.setEnabled(not busy)
        self.save_btn.setEnabled(not busy)

    def saved(self) -> bool:
        return self._saved

    # --- lifecycle: 検証スレッド実行中に閉じられたら待つ ---
    def _await_check(self) -> None:
        th = getattr(self, "_check_thread", None)
        if th is not None and th.isRunning():
            try:
                self._check_worker.report_ready.disconnect()
            except (RuntimeError, TypeError):
                pass
            th.quit()
            th.wait(45000)   # 診断の最大 connect(10s)+read(30s) を上回る値

    def done(self, r: int) -> None:
        self._await_check()
        super().done(r)

    def closeEvent(self, event) -> None:
        self._await_check()
        super().closeEvent(event)
