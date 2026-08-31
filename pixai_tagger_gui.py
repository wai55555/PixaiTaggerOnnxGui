__version__ = "1.5.0"

import sys
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


def main():
    """main entry point."""
    app = QApplication(sys.argv)

    app.setStyle(_FastTooltipStyle("Fusion"))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
