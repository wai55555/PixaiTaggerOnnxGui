from pathlib import Path

from PySide6.QtWidgets import QLineEdit, QListWidget, QWidget
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeyEvent, QWheelEvent
from PySide6.QtCore import Qt, Signal


from utils import write_debug_log, GetString, default_get_string_fallback

class PathLineEdit(QLineEdit):
    # ... (class is unchanged) ...
    folder_dropped = Signal(str)
    def __init__(self, parent: QWidget | None = None, get_string: GetString | None = None):
        super().__init__(parent)
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
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
    def __init__(self, parent: QWidget | None = None, get_string: GetString | None = None):
        super().__init__(parent)
        self.get_string: GetString = get_string if get_string else default_get_string_fallback

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        
        if key in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_H, Qt.Key.Key_K, Qt.Key.Key_Left):
            current = self.currentRow()
            if current > 0:
                self.setCurrentRow(current - 1)
                self.itemClicked.emit(self.currentItem())
                self.scrollToItem(self.currentItem())
                write_debug_log(str(self.get_string("CustomWidgets", "Image_List_Key_Up")), self.get_string)
        
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_J, Qt.Key.Key_L, Qt.Key.Key_Right):
            current = self.currentRow()
            if current < self.count() - 1:
                self.setCurrentRow(current + 1)
                self.itemClicked.emit(self.currentItem())
                self.scrollToItem(self.currentItem())
                write_debug_log(str(self.get_string("CustomWidgets", "Image_List_Key_Down")), self.get_string)
        
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