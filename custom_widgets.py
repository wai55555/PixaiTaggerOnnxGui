from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import QLineEdit, QListWidget
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeyEvent, QWheelEvent
from PySide6.QtCore import Qt, Signal


from utils import write_debug_log

class PathLineEdit(QLineEdit):
    # ... (class is unchanged) ...
    folder_dropped = Signal(str)
    def __init__(self, parent=None, get_string: Callable[[str, str, Any], str] | None = None):
        super().__init__(parent)
        self.get_string = get_string if get_string else lambda section, key, **kwargs: key
        self.setAcceptDrops(True)
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if Path(path).is_dir():
                    event.acceptProposedAction()
                    return
        event.ignore()
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            if Path(path).is_dir():
                self.setText(path)
                self.folder_dropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()


class TagListWidget(QListWidget):
    """A QListWidget that supports keyboard selection and Ctrl + wheel navigation."""
    def __init__(self, *args, get_string: Callable[[str, str, Any], str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_string = get_string if get_string else lambda section, key, **kwargs: key

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        
        if key in (Qt.Key_Up, Qt.Key_W, Qt.Key_A, Qt.Key_H, Qt.Key_K, Qt.Key_Left):
            current = self.currentRow()
            if current > 0:
                self.setCurrentRow(current - 1)
                self.itemClicked.emit(self.currentItem())
                self.scrollToItem(self.currentItem())
                write_debug_log(self.get_string("CustomWidgets", "Image_List_Key_Up"))
        
        elif key in (Qt.Key_Down, Qt.Key_S, Qt.Key_D, Qt.Key_J, Qt.Key_L, Qt.Key_Right):
            current = self.currentRow()
            if current < self.count() - 1:
                self.setCurrentRow(current + 1)
                self.itemClicked.emit(self.currentItem())
                self.scrollToItem(self.currentItem())
                write_debug_log(self.get_string("CustomWidgets", "Image_List_Key_Down"))
        
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """
        Overrides the default wheel event.
        - With Ctrl key: Navigates to the previous/next image.
        - Without Ctrl key: Performs the standard list scrolling.
        """
        # --- MODIFIED: Use bitwise AND for a more robust check ---
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            current = self.currentRow()

            if delta > 0:
                if current > 0:
                    self.setCurrentRow(current - 1)
                    self.itemClicked.emit(self.currentItem())
                    self.scrollToItem(self.currentItem())
            elif delta < 0:
                if current < self.count() - 1:
                    self.setCurrentRow(current + 1)
                    self.itemClicked.emit(self.currentItem())
                    self.scrollToItem(self.currentItem())
            
            event.accept()
        else:
            super().wheelEvent(event)