"""VLM 設定ダイアログ（260901_VLM_design.md 6.3節 / implement_plan 7章）。

通常画面に複雑な HTTP 設定は出さず、ここで管理する:
  - キャプションプロファイル（表示のみ）
  - 実行モード（内蔵の厳格フォールバック / カスタム接続単独）
  - フォールバック経路（内蔵3接続の有効・APIキー・診断）
  - 料金ポリシー（無料経路のみ / 有料継続）
  - 詳細な出力設定（詳細度・文数・キャラクター名・Markdown・最大トークン）
  - カスタム接続の追加・編集・削除
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

import vlm_config
import vlm_models
import vlm_secrets
from app_settings import save_config
from custom_connection_dialog import CustomConnectionDialog
from vlm_connections import ConnectionKind
from vlm_diagnostics import DiagStatus
from vlm_worker import VlmDiagnosticsWorker, VlmModelListWorker

GetString = Callable[..., str]

# 出力設定コンボの選択肢。表示ラベルは locale（[Vlm] Opt_*）から引くので、ここは
# 保存値（config.ini に書く文字列）だけを持つ。
# 出力言語は今は "en" 固定（プロンプトが英語前提）。将来対応言語が増えたらここへ
# 足せば UI に出る。当面はコンボを無効化してグレー表示する。
_LANGUAGE_KEYS = ["en"]
_DETAIL_KEYS = ["standard", "detailed", "maximum_detail"]
_SENTENCE_KEYS = ["automatic_long_detailed", "1", "2", "3", "4", "5"]
_CHARNAME_KEYS = ["do_not_identify", "explicit_only", "allow_guessing"]
_MARKDOWN_KEYS = ["disabled", "allowed"]

_BUILTIN_SECRET_REF = {
    "builtin-gemini": "vlm/gemini/api_key",
    "builtin-openrouter": "vlm/openrouter/api_key",
    "builtin-cloudflare": "vlm/cloudflare/api_token",
    "builtin-groq": "vlm/groq/api_key",
    "builtin-nvidia": "vlm/nvidia/api_key",
    "builtin-mistral": "vlm/mistral/api_key",
}

# 各サービスの「APIキー取得ページ」。key_url が直接キー作成ページ、login_url は
# 未ログイン時に案内する入口。instructions_key は locale の説明文キー。
_PROVIDER_KEY_INFO = {
    "gemini": {
        "key_url": "https://aistudio.google.com/app/api-keys",
        "login_url": "https://aistudio.google.com/",
        "instructions_key": "ApiKey_Steps_Gemini",
    },
    "openrouter": {
        "key_url": "https://openrouter.ai/workspaces/default/keys",
        "login_url": "https://openrouter.ai/",
        "instructions_key": "ApiKey_Steps_OpenRouter",
    },
    "cloudflare": {
        "key_url": "https://dash.cloudflare.com/profile/api-tokens",
        "login_url": "https://dash.cloudflare.com/sign-up",
        "instructions_key": "ApiKey_Steps_Cloudflare",
    },
    "groq": {
        "key_url": "https://console.groq.com/keys",
        "login_url": "https://console.groq.com/login",
        "instructions_key": "ApiKey_Steps_Groq",
    },
    "nvidia": {
        "key_url": "https://build.nvidia.com/settings/api-keys",
        "login_url": "https://build.nvidia.com/login",
        "instructions_key": "ApiKey_Steps_Nvidia",
    },
    "mistral": {
        "key_url": "https://console.mistral.ai/api-keys",
        "login_url": "https://console.mistral.ai/",
        "instructions_key": "ApiKey_Steps_Mistral",
    },
}


class VlmSettingsDialog(QDialog):
    def __init__(self, settings, get_string: GetString, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._vlm = settings.vlm
        self._t = get_string
        self._custom_connections: list[dict] = vlm_config.load_custom_connections()
        self._diag_thread: QThread | None = None
        self._diag_worker: VlmDiagnosticsWorker | None = None
        self.setWindowTitle(get_string("Vlm", "Settings_Title"))
        self.setMinimumWidth(520)
        self._build()
        self._load()

    def _opts(self, prefix: str, keys: list[str]) -> list[tuple[str, str]]:
        """保存値 key と locale から引いた表示ラベルの組にする（[Vlm] <prefix>_<key>）。"""
        return [(k, self._t("Vlm", f"{prefix}_{k}")) for k in keys]

    def _build(self) -> None:
        root = QVBoxLayout(self)

        prof = QGroupBox(self._t("Vlm", "Settings_Profile"))
        pf = QFormLayout(prof)
        prow = QHBoxLayout()
        self.profile_combo = QComboBox()
        prow.addWidget(self.profile_combo, 1)
        self.profile_new_btn = QPushButton(self._t("Vlm", "Profile_New"))
        self.profile_dup_btn = QPushButton(self._t("Vlm", "Profile_Duplicate"))
        self.profile_edit_btn = QPushButton(self._t("Vlm", "Profile_Edit"))
        self.profile_del_btn = QPushButton(self._t("Vlm", "Profile_Delete"))
        self.profile_new_btn.clicked.connect(self._new_profile)
        self.profile_dup_btn.clicked.connect(self._dup_profile)
        self.profile_edit_btn.clicked.connect(self._edit_profile)
        self.profile_del_btn.clicked.connect(self._del_profile)
        for b in (self.profile_new_btn, self.profile_dup_btn, self.profile_edit_btn, self.profile_del_btn):
            prow.addWidget(b)
        pf.addRow(self._t("Vlm", "Settings_Caption_Profile"), prow)
        self.profile_canon_label = QLabel()
        self.profile_canon_label.setStyleSheet("color: gray;")
        self.profile_canon_label.setWordWrap(True)
        pf.addRow("", self.profile_canon_label)
        root.addWidget(prof)
        self._reload_profiles()

        mode = QGroupBox(self._t("Vlm", "Settings_Exec_Mode"))
        mv = QVBoxLayout(mode)
        self.mode_builtin = QRadioButton(self._t("Vlm", "Settings_Mode_Builtin"))
        self.mode_custom = QRadioButton(self._t("Vlm", "Settings_Mode_Custom"))
        row = QHBoxLayout()
        row.addWidget(self.mode_custom)
        self.custom_select = QComboBox()
        row.addWidget(self.custom_select, 1)
        mv.addWidget(self.mode_builtin)
        mv.addLayout(row)
        root.addWidget(mode)

        routes = QGroupBox(self._t("Vlm", "Settings_Routes"))
        rv = QVBoxLayout(routes)
        self._route_rows: dict[str, dict] = {}
        # 経路の優先順位はこのリストの順。▲▼ ボタンで並べ替える。
        self._route_order: list[str] = []
        self._routes_box = QVBoxLayout()
        rv.addLayout(self._routes_box)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        cf_row = QHBoxLayout()
        cf_row.addWidget(QLabel(self._t("Vlm", "Settings_Cf_Account")))
        self.cf_account_edit = QLineEdit()
        cf_row.addWidget(self.cf_account_edit, 1)
        rv.addLayout(cf_row)

        self.strict_check = QCheckBox(self._t("Vlm", "Settings_Strict_Identity"))
        self.strict_check.setToolTip(self._t("Vlm", "Settings_Strict_Identity_Tooltip"))
        rv.addWidget(self.strict_check)
        root.addWidget(routes)

        fee = QGroupBox(self._t("Vlm", "Settings_Fee_Policy"))
        fv = QVBoxLayout(fee)
        self.fee_free_only = QRadioButton(self._t("Vlm", "Settings_Fee_FreeOnly"))
        self.fee_paid = QRadioButton(self._t("Vlm", "Settings_Fee_Paid"))
        fv.addWidget(self.fee_free_only)
        fv.addWidget(self.fee_paid)
        self.fee_note = QLabel(self._t("Vlm", "Settings_Fee_Note"))
        self.fee_note.setWordWrap(True)
        fv.addWidget(self.fee_note)
        root.addWidget(fee)

        det = QGroupBox(self._t("Vlm", "Settings_Detail"))
        dfrm = QFormLayout(det)
        self.detail_combo = _combo(self._opts("Opt_Detail", _DETAIL_KEYS))
        self.sentence_combo = _combo(self._opts("Opt_Sentence", _SENTENCE_KEYS))
        self.charname_combo = _combo(self._opts("Opt_CharName", _CHARNAME_KEYS))
        self.markdown_combo = _combo(self._opts("Opt_Markdown", _MARKDOWN_KEYS))
        self.language_combo = _combo(self._opts("Opt_Language", _LANGUAGE_KEYS))
        # 当面は English 固定。選択式にしておくが操作不可（グレー）にする。
        self.language_combo.setEnabled(len(_LANGUAGE_KEYS) > 1)
        self.language_combo.setToolTip(self._t("Vlm", "Settings_Language_Fixed_Tooltip"))
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(16, 32768)
        dfrm.addRow(self._t("Vlm", "Settings_Language"), self.language_combo)
        dfrm.addRow(self._t("Vlm", "Settings_DetailLevel"), self.detail_combo)
        dfrm.addRow(self._t("Vlm", "Settings_SentenceMode"), self.sentence_combo)
        dfrm.addRow(self._t("Vlm", "Settings_CharName"), self.charname_combo)
        dfrm.addRow(self._t("Vlm", "Settings_Markdown"), self.markdown_combo)
        dfrm.addRow(self._t("Vlm", "Settings_MaxTokens"), self.max_tokens)
        root.addWidget(det)

        cust = QGroupBox(self._t("Vlm", "Settings_Custom"))
        cvv = QVBoxLayout(cust)
        self.custom_list = QListWidget()
        cvv.addWidget(self.custom_list)
        cbtns = QHBoxLayout()
        self.add_custom_btn = QPushButton(self._t("Vlm", "Settings_Custom_Add"))
        self.edit_custom_btn = QPushButton(self._t("Vlm", "Settings_Custom_Edit"))
        self.del_custom_btn = QPushButton(self._t("Vlm", "Settings_Custom_Delete"))
        self.add_custom_btn.clicked.connect(self._add_custom)
        self.edit_custom_btn.clicked.connect(self._edit_custom)
        self.del_custom_btn.clicked.connect(self._delete_custom)
        for b in (self.add_custom_btn, self.edit_custom_btn, self.del_custom_btn):
            cbtns.addWidget(b)
        cvv.addLayout(cbtns)
        root.addWidget(cust)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self._on_save)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)
        root.addWidget(buttons)

        self.mode_custom.toggled.connect(lambda on: self.custom_select.setEnabled(on))
        self.fee_paid.toggled.connect(self._sync_paid_rows)

    def _make_route_row(self, conn) -> dict:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        up = QPushButton("▲")
        down = QPushButton("▼")
        up.setFixedWidth(28)
        down.setFixedWidth(28)
        up.clicked.connect(lambda _=False, cid=conn.connection_id: self._move_route(cid, -1))
        down.clicked.connect(lambda _=False, cid=conn.connection_id: self._move_route(cid, +1))
        enabled = QCheckBox()
        name = QLabel(conn.display_name)
        model_combo = QComboBox()
        model_combo.setEditable(True)
        model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        model_combo.setMinimumWidth(190)
        model_combo.lineEdit().setPlaceholderText(self._t("Vlm", "Settings_Route_ModelId"))
        model_combo.setToolTip(self._t("Vlm", "Settings_Route_ModelId_Tooltip"))
        if conn.model_id:
            model_combo.addItem(conn.model_id)
            model_combo.setCurrentText(conn.model_id)
        else:
            model_combo.setCurrentText("")
        model_combo.lineEdit().editingFinished.connect(
            lambda cid=conn.connection_id: self._on_model_id_edited(cid))
        model_combo.activated.connect(
            lambda _i, cid=conn.connection_id: self._on_model_id_edited(cid))
        list_btn = QPushButton(self._t("Vlm", "Settings_Route_FetchModels"))
        list_btn.setToolTip(self._t("Vlm", "Settings_Route_FetchModels_Tooltip"))
        list_btn.clicked.connect(lambda _=False, cid=conn.connection_id: self._fetch_models(cid))
        register_btn = QPushButton(self._t("Vlm", "Settings_Register_ApiKey"))
        register_btn.clicked.connect(lambda _=False, cid=conn.connection_id: self._open_api_key_dialog(cid))
        status = QLabel()
        paid_ok = QCheckBox(self._t("Vlm", "Settings_Route_PaidOk"))
        diag_btn = QPushButton(self._t("Vlm", "Settings_Diagnose"))
        diag_btn.clicked.connect(lambda _=False, cid=conn.connection_id: self._diagnose_one(cid))
        for w in (up, down, enabled, name, model_combo, list_btn, register_btn, status, paid_ok, diag_btn):
            row.addWidget(w)
        return {
            "widget": widget, "up": up, "down": down, "name": name,
            "enabled": enabled, "model_edit": model_combo, "list_btn": list_btn,
            "register": register_btn,
            "status": status, "paid_ok": paid_ok, "diag_btn": diag_btn,
            "secret_ref": _BUILTIN_SECRET_REF.get(conn.connection_id, conn.auth.secret_ref),
            "conn": conn,
        }

    def _rebuild_routes(self) -> None:
        """内蔵経路の行を作り直す。全プロバイダーを出し、選択プロファイルに binding が
        あるものはその model_id を、無いものは空欄（＝ここに実 ID を入れて接続を試す）。"""
        for r in self._route_rows.values():
            r["widget"].setParent(None)
            r["widget"].deleteLater()
        self._route_rows.clear()
        self._route_order.clear()
        profile = vlm_config.resolve_model_profile(self._vlm)
        if hasattr(self, "profile_canon_label"):
            self.profile_canon_label.setText(
                self._t("Vlm", "Settings_Profile_Canonical",
                        id=profile.canonical_model_id) if profile is not None else "")
        conns = [c for c in vlm_config.build_connection_map(self._vlm, profile).values()
                 if c.kind is ConnectionKind.BUILTIN]
        order = vlm_config.ordered_builtin_provider_ids(self._vlm, profile)
        conns.sort(key=lambda c: order.index(c.provider_id) if c.provider_id in order else 99)
        for conn in conns:
            row = self._make_route_row(conn)
            # このプロファイルに binding が無い経路は、有効化チェックだけ薄くしておく
            # （モデル ID を入れれば診断はできる）。
            has_binding = profile is None or profile.binding_for(conn.provider_id) is not None
            if not has_binding:
                row["name"].setStyleSheet("color: gray;")
                row["name"].setToolTip(self._t("Vlm", "Settings_Route_No_Binding"))
            self._route_rows[conn.connection_id] = row
            self._route_order.append(conn.connection_id)
        self._relayout_routes()
        self._apply_route_states()

    def _apply_route_states(self) -> None:
        order = set(self._vlm.order_list())
        paid = {p.strip() for p in str(self._vlm.paid_connections).split(",") if p.strip()}
        for cid, r in self._route_rows.items():
            provider = r["conn"].provider_id
            r["enabled"].setChecked(provider in order or not order)
            r["paid_ok"].setChecked(provider in paid)
            self._refresh_route_status(cid)
        self._sync_paid_rows()

    def _on_profile_changed(self) -> None:
        pid = self.profile_combo.currentData()
        is_user = vlm_config.is_user_profile(pid or "")
        self.profile_edit_btn.setEnabled(is_user)
        self.profile_del_btn.setEnabled(is_user)
        if not pid or pid == self._vlm.model_profile_id:
            return
        self._vlm.model_profile_id = pid
        self._rebuild_routes()

    def _reload_profiles(self, select_id: str | None = None) -> None:
        want = select_id or self.profile_combo.currentData() or self._vlm.model_profile_id
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for p in vlm_config.all_profiles():
            self.profile_combo.addItem(p.display_name, p.profile_id)
        i = self.profile_combo.findData(want)
        self.profile_combo.setCurrentIndex(i if i >= 0 else 0)
        self.profile_combo.blockSignals(False)
        pid = self.profile_combo.currentData() or ""
        is_user = vlm_config.is_user_profile(pid)
        self.profile_edit_btn.setEnabled(is_user)
        self.profile_del_btn.setEnabled(is_user)

    def _profile_dict_for(self, profile_id: str) -> dict:
        p = next((x for x in vlm_config.all_profiles() if x.profile_id == profile_id), None)
        if p is None:
            return {}
        return {
            "profile_id": p.profile_id, "display_name": p.display_name,
            "canonical_model_id": p.canonical_model_id,
            "bindings": {prov: {"model_id": b.model_id, "free_route": b.free_route}
                         for prov, b in p.bindings.items()},
        }

    def _open_profile_editor(self, src: dict | None, *, read_only: bool = False) -> None:
        from vlm_profile_editor import ProfileEditorDialog
        dlg = ProfileEditorDialog(self._t, src, read_only=read_only, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_profile() is None:
            return
        result = dlg.result_profile()
        users = [d for d in vlm_config.load_user_profiles() if d.get("profile_id") != result["profile_id"]]
        users.append(result)
        vlm_config.save_user_profiles(users)
        self._vlm.model_profile_id = result["profile_id"]
        self._reload_profiles(result["profile_id"])
        self._rebuild_routes()

    def _new_profile(self) -> None:
        self._open_profile_editor(None)

    def _dup_profile(self) -> None:
        src = self._profile_dict_for(self.profile_combo.currentData() or "")
        if src:
            src.pop("profile_id", None)   # 新規 ID を振らせる
            src["display_name"] = f'{src.get("display_name", "")} (copy)'
        self._open_profile_editor(src)

    def _edit_profile(self) -> None:
        pid = self.profile_combo.currentData() or ""
        if vlm_config.is_user_profile(pid):
            self._open_profile_editor(self._profile_dict_for(pid))

    def _del_profile(self) -> None:
        pid = self.profile_combo.currentData() or ""
        if not vlm_config.is_user_profile(pid):
            return
        if QMessageBox.question(self, self._t("Vlm", "Settings_Profile"),
                                self._t("Vlm", "Profile_Delete_Confirm")) != QMessageBox.StandardButton.Yes:
            return
        users = [d for d in vlm_config.load_user_profiles() if d.get("profile_id") != pid]
        vlm_config.save_user_profiles(users)
        self._reload_profiles(vlm_config.all_profiles()[0].profile_id if vlm_config.all_profiles() else None)
        self._vlm.model_profile_id = self.profile_combo.currentData() or self._vlm.model_profile_id
        self._rebuild_routes()

    def _on_model_id_edited(self, cid: str) -> None:
        r = self._route_rows.get(cid)
        if r is None:
            return
        text = r["model_edit"].currentText().strip()
        vlm_config.set_model_id_override(self._vlm, r["conn"].provider_id, text,
                                         profile_id=self._vlm.model_profile_id)
        if text:
            r["conn"].model_id = text   # 診断・キー登録がこの場で新IDを使えるように
        # 「同一モデルを多プロバイダーで回す」前提なので、プロファイルと明らかに別物の
        # ID を入れたら警告する（強制はしない）。
        profile = vlm_config.resolve_model_profile(self._vlm)
        if text and profile is not None and not vlm_models.looks_same_family(profile, text):
            r["status"].setText(self._t("Vlm", "Settings_Route_ModelId_Mismatch",
                                        profile=profile.display_name))

    # --- モデル一覧の取得 -------------------------------------------------------
    def _fetch_models(self, cid: str) -> None:
        if getattr(self, "_ml_thread", None) is not None:
            return
        r = self._route_rows.get(cid)
        if r is None:
            return
        conn_map = vlm_config.build_connection_map(
            self._vlm, vlm_config.resolve_model_profile(self._vlm))
        conn = conn_map.get(cid)
        if conn is None:
            return
        api_key = vlm_secrets.get_secret(conn.auth.secret_ref) if conn.auth.type != "none" else None
        r["status"].setText(self._t("Vlm", "Settings_Route_FetchModels_Busy"))
        self._ml_pending_cid = cid
        self._ml_thread = QThread(self)
        self._ml_worker = VlmModelListWorker(conn, api_key)
        self._ml_worker.moveToThread(self._ml_thread)
        self._ml_thread.started.connect(self._ml_worker.run)
        self._ml_worker.result_ready.connect(self._on_model_list)
        self._ml_worker.finished.connect(self._ml_thread.quit)
        self._ml_thread.finished.connect(self._ml_cleanup)
        for rr in self._route_rows.values():
            rr["list_btn"].setEnabled(False)
        self._ml_thread.start()

    @Slot(str, object)
    def _on_model_list(self, connection_id: str, result) -> None:
        cid = connection_id or getattr(self, "_ml_pending_cid", "")
        r = self._route_rows.get(cid)
        if r is None:
            return
        if not isinstance(result, list):
            detail = getattr(result, "message", "") or str(result)
            r["status"].setText(self._t("Vlm", "Settings_Route_FetchModels_Fail", detail=detail))
            return

        r["model_ids"] = result
        combo = r["model_edit"]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(result)
        combo.blockSignals(False)

        # フォールバックは「同一モデルを多プロバイダーで回す」設計。一覧の中から
        # 選択プロファイルのモデルに一番合うものを自動で当て、ユーザーに確認させる。
        profile = vlm_config.resolve_model_profile(self._vlm)
        best, score = vlm_models.match_model_id(profile, r["conn"].provider_id, result) \
            if profile is not None else (None, 0.0)
        okmsg = self._t("Vlm", "Settings_Route_FetchModels_Ok", n=len(result))
        if best is not None:
            combo.blockSignals(True)
            combo.setCurrentText(best)
            combo.blockSignals(False)
            self._on_model_id_edited(cid)
            key = "Settings_Route_FetchModels_Exact" if score >= 0.999 \
                else "Settings_Route_FetchModels_Matched"
            r["status"].setText(f"{okmsg} — " + self._t("Vlm", key, id=best))
        else:
            combo.blockSignals(True)
            combo.setCurrentText("")
            combo.blockSignals(False)
            r["status"].setText(f"{okmsg} — " + self._t(
                "Vlm", "Settings_Route_FetchModels_NoMatch",
                profile=(profile.display_name if profile else self._vlm.model_profile_id)))

    def _ml_cleanup(self) -> None:
        if getattr(self, "_ml_worker", None) is not None:
            self._ml_worker.deleteLater()
            self._ml_worker = None
        if getattr(self, "_ml_thread", None) is not None:
            self._ml_thread.deleteLater()
            self._ml_thread = None
        for rr in self._route_rows.values():
            rr["list_btn"].setEnabled(True)

    def _await_ml_thread(self) -> None:
        th = getattr(self, "_ml_thread", None)
        if th is not None and th.isRunning():
            try:
                self._ml_worker.result_ready.disconnect()
            except (RuntimeError, TypeError):
                pass
            th.quit()
            th.wait(30000)

    def _relayout_routes(self) -> None:
        while self._routes_box.count():
            self._routes_box.takeAt(0)
        for pos, cid in enumerate(self._route_order):
            r = self._route_rows[cid]
            self._routes_box.addWidget(r["widget"])
            r["up"].setEnabled(pos > 0)
            r["down"].setEnabled(pos < len(self._route_order) - 1)

    def _move_route(self, cid: str, delta: int) -> None:
        i = self._route_order.index(cid)
        j = i + delta
        if 0 <= j < len(self._route_order):
            self._route_order[i], self._route_order[j] = self._route_order[j], self._route_order[i]
            self._relayout_routes()

    # --- load / save ---
    def _load(self) -> None:
        idx = self.profile_combo.findData(self._vlm.model_profile_id)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.profile_combo.blockSignals(False)
        if idx < 0 and self.profile_combo.count():
            self._vlm.model_profile_id = self.profile_combo.currentData()
        self._rebuild_routes()

        self.mode_builtin.setChecked(self._vlm.execution_mode != "custom_single")
        self.mode_custom.setChecked(self._vlm.execution_mode == "custom_single")
        self._refresh_custom_list()
        self.custom_select.setEnabled(self.mode_custom.isChecked())

        self.fee_free_only.setChecked(self._vlm.free_only)
        self.fee_paid.setChecked(not self._vlm.free_only)
        self.strict_check.setChecked(bool(getattr(self._vlm, "strict_identity", False)))

        _select(self.detail_combo, self._vlm.detail_level)
        _select(self.sentence_combo, self._vlm.sentence_mode)
        _select(self.charname_combo, self._vlm.character_name_mode)
        _select(self.markdown_combo, self._vlm.markdown)
        _select(self.language_combo, self._vlm.language or "en")
        self.max_tokens.setValue(int(self._vlm.max_output_tokens))
        self.cf_account_edit.setText(getattr(self._vlm, "cloudflare_account_id", "") or "")
        # 経路行の有効/有料/状態は _rebuild_routes -> _apply_route_states で反映済み。

    def _refresh_route_status(self, cid: str) -> None:
        r = self._route_rows[cid]
        st = vlm_secrets.secret_status(r["secret_ref"]) if r["secret_ref"] else "missing"
        text = self._t("Vlm", f"Settings_Key_Status_{st}")
        token = f"{self._vlm.model_profile_id}:{r['conn'].provider_id}"
        if token in self._vlm.verified_set():
            text += "  " + self._t("Vlm", "Settings_Route_Verified")
        r["status"].setText(text)

    def _sync_paid_rows(self) -> None:
        paid = self.fee_paid.isChecked()
        for r in self._route_rows.values():
            r["paid_ok"].setEnabled(paid)

    def _refresh_custom_list(self) -> None:
        self.custom_list.clear()
        self.custom_select.clear()
        for c in self._custom_connections:
            label = f'{c.get("display_name", c["connection_id"])}  [{c.get("kind", "?")}]'
            item = QListWidgetItem(label)
            item.setData(1000, c["connection_id"])
            self.custom_list.addItem(item)
            self.custom_select.addItem(label, c["connection_id"])
        if self._vlm.selected_connection_id:
            idx = self.custom_select.findData(self._vlm.selected_connection_id)
            if idx >= 0:
                self.custom_select.setCurrentIndex(idx)

    def _add_custom(self) -> None:
        dlg = CustomConnectionDialog(self._t, None, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_connection():
            self._custom_connections.append(dlg.result_connection())
            self._refresh_custom_list()

    def _edit_custom(self) -> None:
        cid = self._selected_custom_id()
        if cid is None:
            return
        current = next((c for c in self._custom_connections if c["connection_id"] == cid), None)
        if current is None:
            return
        dlg = CustomConnectionDialog(self._t, current, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_connection():
            updated = dlg.result_connection()
            self._custom_connections = [updated if c["connection_id"] == cid else c
                                        for c in self._custom_connections]
            self._refresh_custom_list()

    def _delete_custom(self) -> None:
        cid = self._selected_custom_id()
        if cid is None:
            return
        if QMessageBox.question(self, self._t("Vlm", "Settings_Custom"),
                                self._t("Vlm", "Settings_Custom_Delete_Confirm")) != QMessageBox.StandardButton.Yes:
            return
        self._custom_connections = [c for c in self._custom_connections if c["connection_id"] != cid]
        self._refresh_custom_list()

    def _selected_custom_id(self) -> str | None:
        item = self.custom_list.currentItem()
        return item.data(1000) if item else None

    def _diagnose_one(self, cid: str) -> None:
        # 通信は UI スレッドで行わない（NFR-002）。ボタンを無効化してワーカーへ。
        if getattr(self, "_diag_thread", None) is not None:
            return
        # 未保存の Cloudflare account id をこの診断だけ反映する（保存は Save 時）。
        if self.cf_account_edit.text().strip() != (self._vlm.cloudflare_account_id or ""):
            self._vlm.cloudflare_account_id = self.cf_account_edit.text().strip()
        conn_map = vlm_config.build_connection_map(
            self._vlm, vlm_config.resolve_model_profile(self._vlm))
        conn = conn_map.get(cid)
        if conn is None:
            return
        api_key = vlm_secrets.get_secret(conn.auth.secret_ref) if conn.auth.type != "none" else None

        # report_ready はワーカースレッドから飛ぶ。lambda など QObject でないスロットへ
        # つなぐと Qt が受け手のスレッド親和性を判定できず Direct 接続になり、_show_diag_report
        # （QWidget を作り exec() する）がワーカースレッドで走ってクラッシュする。
        # 必ず QObject のバウンドメソッドへつなぎ、対象 conn は self に持たせる。
        self._diag_pending_conn = conn
        self._diag_thread = QThread(self)
        self._diag_worker = VlmDiagnosticsWorker(conn, api_key)
        self._diag_worker.moveToThread(self._diag_thread)
        self._diag_thread.started.connect(self._diag_worker.run)
        self._diag_worker.report_ready.connect(self._on_diag_report)
        self._diag_worker.finished.connect(self._diag_thread.quit)
        self._diag_thread.finished.connect(self._diag_cleanup)
        self._set_diag_buttons_enabled(False)
        self._diag_thread.start()

    @Slot(object)
    def _on_diag_report(self, report) -> None:
        conn = getattr(self, "_diag_pending_conn", None)
        if conn is None:
            return
        # HTTP 200 かつテキスト抽出まで通ったら、その内蔵 binding は「実際に期待どおり
        # 動いた」＝ VERIFIED として記録する（次回以降 UNKNOWN 出荷でも候補に残る）。
        items = {i.name: i.status for i in report.items}
        full_ok = (items.get("HTTP response") is DiagStatus.PASS
                   and items.get("Caption extraction") is DiagStatus.PASS)
        if full_ok and not getattr(conn, "is_custom", True) and getattr(conn, "provider_id", ""):
            if vlm_config.mark_binding_verified(self._vlm, conn.provider_id):
                save_config(self._settings)
                cid = next((c for c, r in self._route_rows.items()
                            if r["conn"].provider_id == conn.provider_id), None)
                if cid:
                    self._refresh_route_status(cid)
        self._show_diag_report(conn, report)

    def _open_api_key_dialog(self, cid: str) -> None:
        from api_key_dialog import ApiKeyDialog
        conn_map = vlm_config.build_connection_map(
            self._vlm, vlm_config.resolve_model_profile(self._vlm))
        conn = conn_map.get(cid)
        r = self._route_rows.get(cid)
        if conn is None or r is None:
            return
        # 未保存の Cloudflare account id をこのチェックだけ反映する。
        if conn.provider_id == "cloudflare" and self.cf_account_edit.text().strip():
            conn.base_url = conn.base_url.replace("{account_id}", self.cf_account_edit.text().strip())
        info = _PROVIDER_KEY_INFO.get(conn.provider_id, {})
        dlg = ApiKeyDialog(
            self._t,
            display_name=conn.display_name,
            secret_ref=r["secret_ref"] or conn.auth.secret_ref,
            conn=conn,
            key_url=info.get("key_url", ""),
            login_url=info.get("login_url", ""),
            instructions=self._t("Vlm", info.get("instructions_key", "ApiKey_Steps_Generic")),
            parent=self,
        )
        dlg.exec()
        self._refresh_route_status(cid)

    def _set_diag_buttons_enabled(self, enabled: bool) -> None:
        for r in self._route_rows.values():
            r["diag_btn"].setEnabled(enabled)

    def _await_diag_thread(self) -> None:
        # 診断スレッド実行中にダイアログが閉じられたら、スレッド破棄前に終了を待つ
        # （QThread: Destroyed while thread is still running を防ぐ）。診断のタイムアウトは
        # 短く固定してあるので待ち時間は限定的。
        th = getattr(self, "_diag_thread", None)
        if th is not None and th.isRunning():
            try:
                self._diag_worker.report_ready.disconnect()
            except (RuntimeError, TypeError):
                pass
            th.quit()
            th.wait(45000)   # 診断の最大 connect(10s)+read(30s) を上回る値

    def done(self, r: int) -> None:
        # accept() / reject() 双方の通り道。Close ボタンも X も window X もここを通る。
        self._await_diag_thread()
        self._await_ml_thread()
        super().done(r)

    def closeEvent(self, event) -> None:
        self._await_diag_thread()
        self._await_ml_thread()
        super().closeEvent(event)

    def _diag_cleanup(self) -> None:
        if getattr(self, "_diag_worker", None) is not None:
            self._diag_worker.deleteLater()
            self._diag_worker = None
        if getattr(self, "_diag_thread", None) is not None:
            self._diag_thread.deleteLater()
            self._diag_thread = None
        self._diag_pending_conn = None
        self._set_diag_buttons_enabled(True)

    def _show_diag_report(self, conn, report) -> None:
        lines = [f"[{i.status.value}] {i.name}: {i.detail}" for i in report.items]
        icon = {DiagStatus.PASS: QMessageBox.Icon.Information,
                DiagStatus.WARN: QMessageBox.Icon.Warning,
                DiagStatus.FAIL: QMessageBox.Icon.Critical}.get(report.overall, QMessageBox.Icon.Information)
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(self._t("Vlm", "Settings_Diagnose"))
        box.setText(f"{conn.display_name}: {report.overall.value}")
        box.setDetailedText("\n".join(lines))
        box.exec()

    def _on_save(self) -> None:
        v = self._vlm
        v.model_profile_id = self.profile_combo.currentData() or v.model_profile_id
        v.execution_mode = "custom_single" if self.mode_custom.isChecked() else "builtin_fallback"
        v.selected_connection_id = self.custom_select.currentData() or "" if self.mode_custom.isChecked() else v.selected_connection_id
        v.free_only = self.fee_free_only.isChecked()
        v.paid_continuation = self.fee_paid.isChecked()
        v.strict_identity = self.strict_check.isChecked()
        v.detail_level = self.detail_combo.currentData()
        v.sentence_mode = self.sentence_combo.currentData()
        v.character_name_mode = self.charname_combo.currentData()
        v.markdown = self.markdown_combo.currentData()
        v.language = self.language_combo.currentData() or "en"
        v.max_output_tokens = int(self.max_tokens.value())
        v.cloudflare_account_id = self.cf_account_edit.text().strip()

        # 経路の順序＝有効集合。▲▼ で決めた self._route_order のうち、有効チェックが
        # 入っている provider だけを、その順で connection_order に書く。
        enabled_providers = [self._route_rows[cid]["conn"].provider_id
                             for cid in self._route_order
                             if self._route_rows[cid]["enabled"].isChecked()]
        if enabled_providers:
            v.connection_order = ",".join(enabled_providers)
        v.paid_connections = ",".join(
            self._route_rows[cid]["conn"].provider_id for cid in self._route_order
            if self._route_rows[cid]["paid_ok"].isChecked())
        # API キーは「APIキー登録」ボタン経由で即時保存されるので、ここでは扱わない。

        vlm_config.save_custom_connections(self._custom_connections)
        try:
            save_config(self._settings)
        except Exception:  # noqa: BLE001
            pass
        self.accept()


def _combo(pairs) -> QComboBox:
    c = QComboBox()
    for value, label in pairs:
        c.addItem(label, value)
    return c


def _select(combo: QComboBox, value) -> None:
    idx = combo.findData(str(value))
    combo.setCurrentIndex(idx if idx >= 0 else 0)
