from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGridLayout, QSizePolicy, QScrollArea, QFrame, QToolTip,
    QApplication
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QWheelEvent, QResizeEvent

import tag_utils
from locale_manager import LocaleManager
from app_settings import AppSettings
from custom_dialogs import ClickableLabel, ImageViewerDialog

TAGS_PER_PAGE_GRID = 14

class ImageEditCellWidget(QWidget):
    # --- ADDED: Signal to request image enlargement with its global index ---
    image_enlarge_requested = Signal(int)

from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGridLayout, QSizePolicy, QScrollArea, QFrame, QToolTip,
    QApplication
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QWheelEvent, QResizeEvent

import tag_utils
from locale_manager import LocaleManager
from app_settings import AppSettings
from custom_dialogs import ClickableLabel, ImageViewerDialog

TAGS_PER_PAGE_GRID = 14

class ImageEditCellWidget(QWidget):
    # --- ADDED: Signal to request image enlargement with its global index ---
    image_enlarge_requested = Signal(int)

    def __init__(self, locale_manager: LocaleManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self._image_path: Path | None = None
        self.tag_buttons: List[QPushButton] = []
        self._current_tag_page = 0
        self._global_index = -1  # To store the image's index in the main list
        
        self.tag_translation_map: dict[str, str] = {}
        self._tag_display_language: str = "English"

        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.image_label = ClickableLabel()
        self.image_label.setText(self.locale_manager.get_string("GridView", "No_Image"))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setStyleSheet("border: 1px solid grey;")
        # --- MODIFIED: Connect to a slot that emits the new signal ---
        self.image_label.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.image_label, 1)

        # ... (rest of the __init__ method is unchanged) ...
        tag_area_widget = QWidget()
        tag_layout = QVBoxLayout(tag_area_widget)
        tag_layout.setContentsMargins(0, 0, 0, 0)
        tag_layout.setSpacing(2)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.tag_grid_layout = QGridLayout(scroll_widget)
        self.tag_grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for i in range(2): self.tag_grid_layout.setColumnStretch(i, 1)
        for i in range(7): self.tag_grid_layout.setRowStretch(i, 1)
        scroll_area.setWidget(scroll_widget)
        tag_layout.addWidget(scroll_area, 1)
        tag_pagination_layout = QHBoxLayout()
        self.prev_tag_page_btn = QPushButton(self.locale_manager.get_string("GridView", "Previous_Page"))
        self.prev_tag_page_btn.clicked.connect(self._prev_tag_page)
        self.next_tag_page_btn = QPushButton(self.locale_manager.get_string("GridView", "Next_Page"))
        self.next_tag_page_btn.clicked.connect(self._next_tag_page)
        tag_pagination_layout.addWidget(self.prev_tag_page_btn)
        tag_pagination_layout.addWidget(self.next_tag_page_btn)
        tag_layout.addLayout(tag_pagination_layout)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        tag_layout.addWidget(separator)
        add_tag_layout = QHBoxLayout()
        self.add_tag_line = QLineEdit()
        self.add_tag_line.setPlaceholderText(self.locale_manager.get_string("GridView", "Add_Tags_Placeholder"))
        self.add_tag_line.returnPressed.connect(self._add_tag)
        add_button = QPushButton(self.locale_manager.get_string("GridView", "Add_Button"))
        add_button.clicked.connect(self._add_tag)
        add_tag_layout.addWidget(self.add_tag_line)
        add_tag_layout.addWidget(add_button)
        tag_layout.addLayout(add_tag_layout)
        layout.addWidget(tag_area_widget, 1)

    # --- ADDED: Slot to handle the double click and emit the request ---
    @Slot()
    def _on_double_click(self):
        if self._global_index != -1:
            self.image_enlarge_requested.emit(self._global_index)

    # --- MODIFIED: load_data now accepts the global index ---
    def load_data(self, image_path: Path, global_index: int):
        self._image_path = image_path
        self._global_index = global_index
        self._current_tag_page = 0
        self._update_tag_display()

    # ... (rest of the class is unchanged) ...
    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self._update_tag_display()
    def _update_tag_display(self):
        if self._image_path and self.image_label.width() > 0 and self.image_label.height() > 0:
            pixmap = QPixmap(str(self._image_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.image_label.size(), 
                                              Qt.AspectRatioMode.KeepAspectRatio, 
                                              Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
        for btn in self.tag_buttons:
            btn.deleteLater()
        self.tag_buttons.clear()
        if not self._image_path: 
            self.prev_tag_page_btn.setEnabled(False)
            self.next_tag_page_btn.setEnabled(False)
            return
        txt_path = tag_utils.get_txt_path(self._image_path)
        tags = tag_utils.read_tags(txt_path)
        total_tags = len(tags)
        start_index = self._current_tag_page * TAGS_PER_PAGE_GRID
        end_index = min(start_index + TAGS_PER_PAGE_GRID, total_tags)
        current_page_tags = tags[start_index:end_index]
        for i, tag in enumerate(current_page_tags):
            display_text = tag
            if self._tag_display_language == "日本語":
                display_text = self.tag_translation_map.get(tag, tag)
            
            btn = QPushButton(display_text)
            btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            btn.setStyleSheet("font-size: 11pt; text-align: left; padding-left: 3px;")
            btn.setToolTip(tag) # Tooltip always shows English tag
            btn.setProperty("original_tag", tag)
            btn.clicked.connect(lambda checked=False, t=tag: self._remove_tag(t))
            row = i % 7; col = i // 7
            self.tag_grid_layout.addWidget(btn, row, col)
            self.tag_buttons.append(btn)
        self.prev_tag_page_btn.setEnabled(self._current_tag_page > 0)
        self.next_tag_page_btn.setEnabled(end_index < total_tags)
    @Slot()
    def _add_tag(self):
        if not self._image_path:
            return
        tags_to_add_str = self.add_tag_line.text().strip()
        if not tags_to_add_str: return
        tags_to_add = [t.strip() for t in tags_to_add_str.split(',') if t.strip()]
        if not tags_to_add: return
        txt_path = tag_utils.get_txt_path(self._image_path)
        existing_tags = tag_utils.read_tags(txt_path)
        actually_new_tags: list[str] = []; any_duplicates = False
        for tag in tags_to_add:
            if tag in existing_tags: any_duplicates = True
            else:
                if tag not in actually_new_tags: actually_new_tags.append(tag)
        self.add_tag_line.clear()
        if any_duplicates:
            tooltip_message = self.locale_manager.get_string("GridView", "Tooltip_Duplication")
            tooltip_pos = self.add_tag_line.mapToGlobal(self.add_tag_line.rect().bottomLeft())
            QToolTip.showText(tooltip_pos, tooltip_message, self.add_tag_line)
        if actually_new_tags:
            if tag_utils.add_tags_to_file(txt_path, actually_new_tags): self._update_tag_display()
    @Slot(str)
    def _remove_tag(self, tag_to_remove: str):
        if not self._image_path: return
        txt_path = tag_utils.get_txt_path(self._image_path)
        if tag_utils.remove_tag_from_file(txt_path, tag_to_remove):
            tags = tag_utils.read_tags(txt_path)
            total_pages = (len(tags) + TAGS_PER_PAGE_GRID - 1) // TAGS_PER_PAGE_GRID
            total_pages = max(1, total_pages)
            if self._current_tag_page >= total_pages: self._current_tag_page = max(0, total_pages - 1)
            self._update_tag_display()
    @Slot()
    def _prev_tag_page(self):
        if self._current_tag_page > 0:
            self._current_tag_page -= 1
            self._update_tag_display()
    @Slot()
    def _next_tag_page(self):
        if not self._image_path: return
        txt_path = tag_utils.get_txt_path(self._image_path)
        tags = tag_utils.read_tags(txt_path)
        if (self._current_tag_page + 1) * TAGS_PER_PAGE_GRID < len(tags):
            self._current_tag_page += 1
            self._update_tag_display()
    def clear_data(self):
        self._image_path = None
        self._global_index = -1
        self._current_tag_page = 0
        self.image_label.clear()
        self.image_label.setText(self.locale_manager.get_string("GridView", "No_Image"))
        for btn in self.tag_buttons: btn.deleteLater()
        self.tag_buttons.clear()
        self.prev_tag_page_btn.setEnabled(False)
        self.next_tag_page_btn.setEnabled(False)

    def set_tag_display_language(self, language: str, translation_map: dict[str, str]):
        self._tag_display_language = language
        self.tag_translation_map = translation_map
        self._update_tag_display()

class GridViewWidget(QWidget):
    back_to_main_requested = Signal()
    
    def __init__(self, settings: AppSettings, locale_manager: LocaleManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.settings = settings
        self.locale_manager = locale_manager
        self._image_paths: List[Path] = []
        self._current_page = 0
        self.cells: List[ImageEditCellWidget] = []
        
        # --- ADDED: State management for the dialog ---
        self._image_viewer_dialog: ImageViewerDialog | None = None
        self._current_dialog_image_index = -1

        self.initUI()
    
    def initUI(self):
        # ... (initUI layout setup is unchanged) ...
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        top_bar_layout = QHBoxLayout()
        self.back_button = QPushButton(self.locale_manager.get_string("GridView", "Back_To_Main_View"))
        self.back_button.clicked.connect(self.back_to_main_requested.emit)
        top_bar_layout.addStretch(1)
        top_bar_layout.addWidget(self.back_button)
        main_layout.addLayout(top_bar_layout)
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(2)
        main_layout.addLayout(self.grid_layout, 1)

        for i in range(9):
            cell = ImageEditCellWidget(self.locale_manager)
            # --- ADDED: Connect the cell's request signal to the handler slot ---
            cell.image_enlarge_requested.connect(self.show_enlarged_image_at_index)
            self.cells.append(cell)
            row, col = divmod(i, 3)
            self.grid_layout.addWidget(cell, row, col)
            
        # ... (initUI pagination setup is unchanged) ...
        pagination_layout = QHBoxLayout()
        self.prev_page_btn = QPushButton(self.locale_manager.get_string("GridView", "Previous_9"))
        self.prev_page_btn.clicked.connect(self.prev_page)
        self.prev_page_btn.setMinimumHeight(40)
        self.prev_page_btn.setStyleSheet("font-size: 14pt;")
        self.next_page_btn = QPushButton(self.locale_manager.get_string("GridView", "Next_9"))
        self.next_page_btn.clicked.connect(self.next_page)
        self.next_page_btn.setMinimumHeight(40)
        self.next_page_btn.setStyleSheet("font-size: 14pt;")
        self.page_label = QLabel("Page 1 / 1")
        pagination_layout.addStretch(2)
        pagination_layout.addWidget(self.prev_page_btn)
        pagination_layout.addStretch(1)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addStretch(1)
        pagination_layout.addWidget(self.next_page_btn)
        pagination_layout.addStretch(2)
        main_layout.addLayout(pagination_layout)

    def _display_page(self):
        start_index = self._current_page * 9
        page_paths = self._image_paths[start_index : start_index + 9]

        for i, cell in enumerate(self.cells):
            if i < len(page_paths):
                # --- MODIFIED: Pass the global index to the cell ---
                global_index = start_index + i
                cell.load_data(page_paths[i], global_index)
                cell.show()
            else:
                cell.clear_data()
                cell.hide()
        
        self._update_pagination_controls()

    # --- ADDED: Centralized handler for showing and managing the dialog ---
    @Slot(int)
    def show_enlarged_image_at_index(self, index: int):
        self._current_dialog_image_index = index

        if self._image_viewer_dialog is None:
            self._image_viewer_dialog = ImageViewerDialog(self, zoom_and_pan_enabled=True)
            self._image_viewer_dialog.nextImageRequested.connect(self._navigate_dialog_image_next)
            self._image_viewer_dialog.prevImageRequested.connect(self._navigate_dialog_image_prev)
            self._image_viewer_dialog.finished.connect(self._on_image_viewer_closed)
        
        self._load_image_into_dialog(index)

        if not self._image_viewer_dialog.isVisible():
            self._image_viewer_dialog.show()
            self._image_viewer_dialog.activateWindow()

    # --- ADDED: Helper method to load/update the image in the dialog ---
    def _load_image_into_dialog(self, index: int):
        if not (0 <= index < len(self._image_paths)) or not self._image_viewer_dialog:
            return

        path = self._image_paths[index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return

        # Set initial size and position only if it's the first time showing
        if not self._image_viewer_dialog.isVisible():
            screen_rect = QApplication.primaryScreen().availableGeometry()
            scaled_size = pixmap.size().scaled(screen_rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self._image_viewer_dialog.resize(scaled_size)
            self._image_viewer_dialog.move(screen_rect.center() - self._image_viewer_dialog.rect().center())
        
        self._image_viewer_dialog.setPixmap(pixmap)

    # --- ADDED: Slots for handling navigation signals from the dialog ---
    @Slot()
    def _navigate_dialog_image_next(self):
        if self._current_dialog_image_index < len(self._image_paths) - 1:
            self._current_dialog_image_index += 1
            self._load_image_into_dialog(self._current_dialog_image_index)

    @Slot()
    def _navigate_dialog_image_prev(self):
        if self._current_dialog_image_index > 0:
            self._current_dialog_image_index -= 1
            self._load_image_into_dialog(self._current_dialog_image_index)

    # --- ADDED: Cleanup slot for when the dialog is closed ---
    @Slot()
    def _on_image_viewer_closed(self):
        self._image_viewer_dialog = None


    # ... (rest of the file is unchanged) ...
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.prev_page()
            event.accept()
        elif delta < 0:
            self.next_page()
            event.accept()
        else:
            super().wheelEvent(event)
    def load_images(self, image_paths: List[Path]):
        self._image_paths = image_paths
        self._current_page = 0
        self._display_page()
    def prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._display_page()
    def next_page(self):
        if (self._current_page + 1) * 9 < len(self._image_paths):
            self._current_page += 1
            self._display_page()
    def _update_pagination_controls(self):
        total_pages = (len(self._image_paths) + 8) // 9
        total_pages = max(1, total_pages)
        self.page_label.setText(f"Page {self._current_page + 1} / {total_pages}")
        self.prev_page_btn.setEnabled(self._current_page > 0)
        self.next_page_btn.setEnabled((self._current_page + 1) * 9 < len(self._image_paths))

    def set_tag_display_language(self, language: str, translation_map: dict[str, str]):
        for cell in self.cells:
            cell.set_tag_display_language(language, translation_map)