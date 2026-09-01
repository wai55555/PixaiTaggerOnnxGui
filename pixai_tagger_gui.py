__version__ = "1.5.0"

import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle
from main_window import MainWindow


class _FastTooltipStyle(QProxyStyle):
    """Shortens the tooltip hover delay app-wide.

    Fusion's default SH_ToolTip_WakeUpDelay is 700 ms, which feels sluggish for the
    model-picker dropdown where the tooltips carry the per-model descriptions the user
    is deliberately hovering to read. 300 ms stays clear of accidental flicker while
    dragging the pointer across items.
    """

    WAKE_UP_DELAY_MS = 300

    def styleHint(self, hint, option=None, widget=None, return_data=None):
        if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return self.WAKE_UP_DELAY_MS
        return super().styleHint(hint, option, widget, return_data)


def os_prefers_dark(app: QApplication) -> bool:
    """OS のカラースキームがダークかどうかを返す。

    アプリは Fusion スタイルを強制しており、Fusion は OS のテーマに追随しない固定
    パレットを持つ。そのため `QApplication.palette()` を見てもライト判定にしかならず、
    「システムテーマ感知が働かない」状態になっていた。Qt 6.5 以降の
    `QStyleHints.colorScheme()` はスタイルに依存せず OS 設定そのものを返すので、
    そちらを一次情報として使い、取得できない環境ではパレットの明度に戻す。
    """
    try:
        scheme = app.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except Exception:
        pass
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def apply_dark_palette(app: QApplication) -> None:
    """Fusion 用のダークパレットを適用する。

    検出だけ直してもウィンドウ自体はライトのままなので、ログ色やハイライト色だけが
    ダーク向けになってちぐはぐになる。OS がダークなら見た目もダークに揃える。
    """
    p = QPalette()
    window = QColor(53, 53, 53)
    base = QColor(35, 35, 35)
    text = QColor(220, 220, 220)
    disabled = QColor(127, 127, 127)
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, window)
    p.setColor(QPalette.ColorRole.ToolTipBase, window)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, window)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    p.setColor(QPalette.ColorRole.PlaceholderText, disabled)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.HighlightedText):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    app.setPalette(p)


def main():
    """main entry point."""
    app = QApplication(sys.argv)

    app.setStyle(_FastTooltipStyle("Fusion"))

    # スタイルを適用した後に OS のテーマへ合わせる。MainWindow はこの時点のパレットから
    # ダーク判定を行うので、ウィンドウ生成より前に済ませておく必要がある。
    if os_prefers_dark(app):
        apply_dark_palette(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
