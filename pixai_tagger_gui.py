__version__ = "0.1.0"

import sys
from PySide6.QtWidgets import QApplication, QLineEdit
from PySide6.QtCore import QEvent, Qt, QObject
from PySide6.QtGui import QKeyEvent
from main_window import MainWindow

class CustomApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.main_window: MainWindow | None = None

    def notify(self, receiver: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            key_event = event
            if key_event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if key_event.key() == Qt.Key.Key_Up or key_event.key() == Qt.Key.Key_Down:
                    if key_event.isAutoRepeat():
                        return False # 自動リピートイベントは無視

                    # 現在フォーカスを持っているウィジェットが QLineEdit のインスタンスである場合のみ処理
                    if isinstance(self.focusWidget(), QLineEdit):
                        if self.main_window:
                            delta = -1 if key_event.key() == Qt.Key.Key_Up else 1
                            self.main_window._navigate_image_list(delta)
                            return True # イベントを処理したので伝播を停止
        
        return super().notify(receiver, event)

def main():
    """main entry point."""
    app = CustomApplication(sys.argv) # CustomApplication を使用

    app.setStyle('Fusion')
    
    window = MainWindow()
    app.main_window = window # MainWindow のインスタンスを CustomApplication に渡す
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
