from pathlib import Path
from typing import Any, Mapping
from datetime import datetime
import locale
import functools

from PySide6.QtCore import (
    Qt, QThread, QObject, Signal, Slot, QTimer, QPoint, QRect, QEvent, QEventLoop
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget,
    QGridLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QTextEdit, QFileDialog, QMessageBox,
    QStackedWidget, QApplication, QSplitter, QListWidgetItem
)
from PySide6.QtGui import (
    QPixmap, QImage, QKeyEvent, QResizeEvent, QDragEnterEvent,
    QDropEvent, QCloseEvent, QWheelEvent,
    QPalette
)

from utils import write_debug_log
import constants
import app_settings # Added import
from app_settings import load_config, load_settings, save_config # Updated import
from custom_widgets import PathLineEdit, TagListWidget
from custom_dialogs import ClickableLabel, ImageViewerDialog
from grid_view_widget import GridViewWidget
from workers import DownloaderWorker, TaggerThreadWorker, TagLoader, BulkTagWorker
from locale_manager import LocaleManager
from ui_main_window import Ui_MainWindow

# --- Main Window ---

class MainWindow(QMainWindow):
    """Main application window for image tagging."""
    # --- UI Elements ---
    central_widget: QStackedWidget
    main_view_widget: QWidget
    grid_view_widget: GridViewWidget
    splitter: QSplitter
    image_list: TagListWidget
    right_vertical_splitter: QSplitter
    input_line: PathLineEdit
    grid_view_button: QPushButton
    image_label: ClickableLabel
    tag_display_grid: QGridLayout
    image_tag_prev_page_btn: QPushButton
    image_tag_next_page_btn: QPushButton
    add_single_tag_line: QLineEdit
    add_single_tag_button: QPushButton
    run_button: QPushButton
    loading_label: QLabel
    tag_button_grid: QGridLayout
    prev_page_btn: QPushButton
    next_page_btn: QPushButton
    add_tag_line: QLineEdit
    add_tag_button: QPushButton
    add_tag_line_append: QLineEdit
    add_tag_button_append: QPushButton
    log_output: QTextEdit

    # --- Signals ---
    request_overwrite_check = Signal(str, str)
    overwrite_dialog_requested = Signal(Path) 
    _overwrite_request = Signal(Path) 
    _request_worker_stop = Signal() # Signal to request workers to stop
    
    def __init__(self):
        super().__init__()
        
        self._initialize_settings_and_locale()
        self._initialize_state()
        
        self.ui = Ui_MainWindow()
        self.ui.setup_ui(self)

        write_debug_log(self.locale_manager.get_string("MainWindow", "MainWindow_Init_Complete"))

        QTimer.singleShot(0, self.initial_load)

    def _initialize_settings_and_locale(self):
        """Loads configuration and initializes the localization manager."""
        config = load_config()
        self.settings = load_settings(config)

        if not self.settings.language_code:
            default_locale = locale.getdefaultlocale()[0]
            os_lang = default_locale.split('_')[0] if default_locale else "en"
            self.settings.language_code = os_lang
            save_config(self.settings)

        self.locale_manager = LocaleManager(self.settings.language_code, constants.BASE_DIR)
        app_settings.set_get_string_func(self.locale_manager.get_string) # Add this line
        write_debug_log(self.locale_manager.get_string("MainWindow", "Application_Startup"))

        self._is_dark_theme = QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128
        self._log_color_map = self._get_log_color_map()
        
    def _initialize_state(self):
        """Initializes all state variables for the main window."""
        # Thread and worker management
        self._tagger_thread: QThread | None = None
        self._tagger_worker: TaggerThreadWorker | None = None
        self._download_thread: QThread | None = None
        self._downloader_worker: DownloaderWorker | None = None
        self._bulk_tag_thread: QThread | None = None
        self._bulk_tag_worker: BulkTagWorker | None = None
        self.tag_thread: QThread | None = None
        self.tag_worker: TagLoader | None = None

        # UI State variables
        self._all_tags: list[tuple[str, int]] = []
        self._current_page: int = 0  # For bulk tag view
        self._current_image_tags: list[str] = []
        self._current_image_tag_page: int = 0
        self._original_image_pixmap: QPixmap | None = None
        self.tag_buttons: list[QPushButton] = []
        self.tag_buttons_for_image: list[QPushButton] = []

        # Processing State variables
        self._is_downloading = False
        self._is_bulk_deleting = False 
        self._always_overwrite: bool = False
        self._always_skip: bool = False
        
        # Timers and Dialogs
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(100)
        self.loading_timer: QTimer | None = None
        self.loading_state = 0
        self._image_viewer_dialog: ImageViewerDialog | None = None
        
        self._overwrite_event_loop: QEventLoop | None = None
        self._overwrite_response: bool | None = None
        self._worker_finished_event_loop: QEventLoop | None = None
        
        # Constants
        self._tag_button_min_width = 100
        self._tag_button_min_height = 25

    def initial_load(self):
        """Performs the initial loading of images and tags after the main window is shown."""
        self.reload_image_list()
        self.reload_tags_only()

    def reload_image_list(self, auto_select_path: str | None = None):
        """Reloads the list of images from the input directory."""
        self.image_label.setPixmap(QPixmap()) # Explicitly clear pixmap
        input_dir_path = Path(self.settings.paths.input_dir)
        self.image_list.clear()
        self.image_label.setText(self.locale_manager.get_string("MainWindow", "Loading_Image_List"))

        if not self._ensure_input_dir_exists(input_dir_path):
            return

        write_debug_log(self.locale_manager.get_string("MainWindow", "Reloading_Image_List", input_dir_path=input_dir_path))
        image_paths = self._get_image_paths(input_dir_path)

        if not image_paths:
            self.image_label.setText(self.locale_manager.get_string("MainWindow", "Image_Files_Not_Found"))
            self.update_log(self.locale_manager.get_string("MainWindow", "Warning_No_Image_Files_Found", input_dir_path_name=input_dir_path.name), "orange")
            return
            
        selected_item = self._populate_image_list(image_paths, auto_select_path)
            
        if not selected_item and self.image_list.count() > 0:
            selected_item = self.image_list.item(0)
        
        if selected_item:
            self.image_list.setCurrentItem(selected_item)
            # Schedule the image loading to ensure the widget is sized correctly
            QTimer.singleShot(100, lambda: self._load_and_fit_image(selected_item))
        else:
            self._clear_image_display()

        self.update_log(self.locale_manager.get_string("MainWindow", "List_Updated_Total_Images", count=len(image_paths)), "blue")

    def _ensure_input_dir_exists(self, path: Path) -> bool:
        """Checks if the input directory exists, creating it if necessary. Returns True on success."""
        if not path.is_dir():
            try:
                path.mkdir(parents=True, exist_ok=True)
                self.update_log(self.locale_manager.get_string("MainWindow", "Info_Input_Folder_Created", input_dir_path_name=path.name), "blue")
                return True
            except Exception as e:
                self.update_log(self.locale_manager.get_string("MainWindow", "Error_Input_Folder_Creation_Failed", input_dir_path_name=path.name, e=e), "red")
                self.image_label.setText(self.locale_manager.get_string("MainWindow", "Input_Folder_Not_Found"))
                return False
        return True

    def _get_image_paths(self, base_path: Path) -> list[Path]:
        """Recursively finds all image files in the given directory."""
        paths: list[Path] = []
        for ext in constants.IMAGE_EXTENSIONS:
             paths.extend(base_path.rglob(f"*{ext}")) 
        return sorted(paths)

    def _populate_image_list(self, paths: list[Path], auto_select: str | None) -> QListWidgetItem | None:
        """Adds image paths to the list widget and determines which item to select."""
        selected_item = None
        input_dir = Path(self.settings.paths.input_dir)
        for path in paths:
            relative_path = str(path.relative_to(input_dir))
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole + 1, relative_path)
            self.image_list.addItem(item)
            
            if auto_select and (path.name == auto_select or relative_path == auto_select):
                 selected_item = item
        
        if not selected_item and self.image_list.count() > 0:
            selected_item = self.image_list.item(0)
        
        return selected_item

    def reload_tags_only(self):
        """Reloads the aggregated tag list for bulk editing asynchronously."""
        if self.tag_thread and self.tag_thread.isRunning():
            # Stop the existing thread gracefully
            if self.tag_worker:
                self.tag_worker.stop()
            self.tag_thread.quit()
            self.tag_thread.wait(1000) # Wait up to 1 second for the thread to finish
            if self.tag_thread.isRunning():
                self.tag_thread.terminate() # Force terminate if it doesn't stop
            
            # Clean up old thread and worker
            if self.tag_worker:
                self.tag_worker.deleteLater()
            self.tag_thread.deleteLater()
            self.tag_thread = None
            self.tag_worker = None
        
        input_dir_path = Path(self.settings.paths.input_dir)
        if not input_dir_path.is_dir():
            return

        self.loading_label.setText(self.locale_manager.get_string("Constants", "Loading_Tag_List"))
        self._set_bulk_controls_enabled(False)
        
        self.tag_thread = QThread()
        self.tag_worker = TagLoader(input_dir_path, self.locale_manager.get_string)
        self.tag_worker.moveToThread(self.tag_thread)
        
        self.tag_thread.started.connect(self.tag_worker.run)
        self.tag_worker.tags_loaded.connect(self._update_bulk_tag_buttons)
        self.tag_worker.finished.connect(self._on_tag_loader_finished)
        
        if self.loading_timer is None:
            self.loading_timer = QTimer(self)
            self.loading_timer.timeout.connect(self._animate_loading_label)

        self.tag_thread.start()
        self.loading_timer.start(100)
    
    def _load_and_fit_image(self, item: QListWidgetItem | None):
        """Loads the image and its tags for the selected list item."""
        if item is None:
            self._clear_image_display()
            return

        image_path = Path(self.settings.paths.input_dir) / item.data(Qt.ItemDataRole.UserRole + 1)
        
        if not image_path.is_file():
            self._clear_image_display()
            self.image_label.setText(self.locale_manager.get_string("MainWindow", "File_Not_Found", image_relative_path=image_path.name))
            return
        
        try:
            image = QImage(str(image_path))
            if image.isNull():
                raise ValueError("Failed to load QImage")
            
            pixmap = QPixmap.fromImage(image)
            self._original_image_pixmap = pixmap
            
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            self._load_image_tags(image_path)
            
            if self._image_viewer_dialog and self._image_viewer_dialog.isVisible():
                self.show_enlarged_image()
        except Exception as e:
            write_debug_log(self.locale_manager.get_string("MainWindow", "Image_Load_Error_Debug", image_path=image_path, e=e))
            self._clear_image_display()
            self.image_label.setText(self.locale_manager.get_string("MainWindow", "Image_Display_Error", image_relative_path=image_path.name, e=e))

    def _load_image_tags(self, image_path: Path):
        """Loads tags from the corresponding .txt file for a given image."""
        txt_path = image_path.with_suffix('.txt')
        tag_content = ""
        if txt_path.is_file():
            try:
                tag_content = txt_path.read_text(encoding='utf-8').strip()
            except Exception as e:
                self.update_log(self.locale_manager.get_string("MainWindow", "Error_Tag_File_Load_Failed", e=e, txt_path_name=txt_path.name), "red")
        
        self._current_image_tags = [tag.strip() for tag in tag_content.split(',') if tag.strip()]
        self._current_image_tag_page = 0
        self._display_image_tag_page()

    def _clear_image_display(self):
        """Clears the image viewer and tag display."""
        self.image_label.clear()
        self.image_label.setText(self.locale_manager.get_string("MainWindow", "Image_Not_Selected"))
        self._original_image_pixmap = None
        self._current_image_tags = []
        self._current_image_tag_page = 0
        self._display_image_tag_page()
        if self._image_viewer_dialog:
            self._image_viewer_dialog.close()

    def _update_bulk_tag_buttons(self, all_tags: list[tuple[str, int]]):
        """Updates the bulk tag deletion buttons with new tag data."""
        if self.loading_timer:
            self.loading_timer.stop()
        self._is_bulk_deleting = False
        self._all_tags = all_tags
        self._current_page = 0
        self.display_current_tag_page()

    def display_current_tag_page(self):
        """Displays the current page of bulk tags."""
        for button in self.tag_buttons:
            button.deleteLater()
        self.tag_buttons.clear()

        total_tags = len(self._all_tags)
        start_index = self._current_page * constants.TAGS_PER_PAGE
        end_index = min(start_index + constants.TAGS_PER_PAGE, total_tags)
        
        if total_tags == 0:
            self.loading_label.setText(self.locale_manager.get_string("MainWindow", "Tag_File_Txt_Not_Found"))
        else:
            self.loading_label.setText(self.locale_manager.get_string("MainWindow", "Displaying_Tags_Count_And_Click_Delete", total_tags=total_tags, start_index=start_index + 1, end_index=end_index))
            
            current_page_tags = self._all_tags[start_index:end_index]
            for i, (tag_name, count) in enumerate(current_page_tags):
                button = QPushButton(f"{tag_name} ({count})")
                button.setMinimumWidth(self._tag_button_min_width)
                button.setFixedHeight(self._tag_button_min_height)
                button.setToolTip(self.locale_manager.get_string("MainWindow", "Tag_Button_Tooltip", tag_name=tag_name))
                button.clicked.connect(functools.partial(self.delete_tag_all, tag_name))
                self.tag_button_grid.addWidget(button, i // 4, i % 4)
                self.tag_buttons.append(button)

        self.prev_page_btn.setEnabled(self._current_page > 0)
        self.next_page_btn.setEnabled(end_index < total_tags)
        self._set_bulk_controls_enabled(True)
        QTimer.singleShot(0, self.update_all_button_alignments)
        
    def _display_image_tag_page(self):
        """Displays the current page of tags for the selected image."""
        for button in self.tag_buttons_for_image:
            button.deleteLater()
        self.tag_buttons_for_image.clear()

        cols = self.settings.window.tag_display_cols
        rows = self.settings.window.tag_display_rows
        tags_per_page = max(1, cols * rows)

        total_tags = len(self._current_image_tags)
        start = self._current_image_tag_page * tags_per_page
        end = min(start + tags_per_page, total_tags)
        
        for i, tag_name in enumerate(self._current_image_tags[start:end]):
            button = QPushButton(tag_name)
            button.setMinimumSize(self._tag_button_min_width, self._tag_button_min_height)
            button.setToolTip(self.locale_manager.get_string("MainWindow", "Tag_Button_Tooltip_Delete", tag_name=tag_name))
            button.clicked.connect(functools.partial(self._delete_image_tag, tag_name))
            self.tag_display_grid.addWidget(button, i // cols, i % cols)
            self.tag_buttons_for_image.append(button)

        self.image_tag_prev_page_btn.setEnabled(self._current_image_tag_page > 0)
        self.image_tag_next_page_btn.setEnabled(end < total_tags)
        QTimer.singleShot(0, self.update_all_button_alignments)

    def _update_ui_for_processing(self, is_running: bool, process_type: str):
        """Updates UI elements based on whether a process is starting or stopping."""
        if is_running:
            style = constants.STYLE_BTN_RED
            if process_type == 'tagging':
                text = self.locale_manager.get_string("Constants", "Stop_Tagging_Process")
            else: # downloading
                text = f"{self.locale_manager.get_string('Constants', 'Stop_Button_Text')} (0%)"
            
            self._set_main_controls_enabled(False)
        else:
            self._set_main_controls_enabled(True)
            self._check_model_status_and_update_ui()
            return

        self.run_button.setStyleSheet(style)
        self.run_button.setText(text)
        self.run_button.setEnabled(True)

    def _set_main_controls_enabled(self, enabled: bool):
        """Enables or disables main UI controls during processing."""
        self.input_line.setEnabled(enabled)
        self.grid_view_button.setEnabled(enabled)
        self.image_list.setEnabled(enabled)
        self._set_bulk_controls_enabled(enabled)

    def _set_bulk_controls_enabled(self, enabled: bool):
        """Enables or disables bulk action controls."""
        for button in self.tag_buttons:
            button.setEnabled(enabled)
        self.add_tag_button.setEnabled(enabled)
        self.add_tag_button_append.setEnabled(enabled)
        self.add_tag_line.setEnabled(enabled)
        self.add_tag_line_append.setEnabled(enabled)

    def closeEvent(self, event: QCloseEvent):
        """Handles the window closing event to save settings and stop threads gracefully."""
        write_debug_log("closeEvent triggered")
        self.save_current_config()

        threads_to_stop: list[tuple[QThread | None, DownloaderWorker | TaggerThreadWorker | TagLoader | BulkTagWorker | None]] = [
            (self._download_thread, self._downloader_worker),
            (self._tagger_thread, self._tagger_worker),
            (self._bulk_tag_thread, self._bulk_tag_worker),
            (self.tag_thread, self.tag_worker)
        ]

        for thread, worker in threads_to_stop:
            if thread and thread.isRunning():
                write_debug_log(f"Requesting thread {thread} to quit.")
                
                if worker is not None and hasattr(worker, 'stop'):
                    self._request_worker_stop.connect(worker.stop, Qt.ConnectionType.QueuedConnection)
                    self._request_worker_stop.emit() # Request worker to stop
                    QApplication.processEvents() # Allow signal to be processed
                    self._request_worker_stop.disconnect(worker.stop) # Disconnect immediately

                thread.quit()
                
                # Wait for the worker's finished signal
                if worker is not None:
                    signal_to_connect: Any | None = None # Changed Signal to Any to resolve Pylance reportAssignmentType
                    if isinstance(worker, DownloaderWorker):
                        signal_to_connect = worker.download_finished
                    elif hasattr(worker, 'finished'):
                        # TaggerThreadWorker, TagLoader, BulkTagWorker should have 'finished'
                        # Type check necessary for Pylance
                        if isinstance(worker, (TaggerThreadWorker, TagLoader, BulkTagWorker)): # type: ignore
                            signal_to_connect = worker.finished
                    
                    if signal_to_connect:
                        loop = QEventLoop()
                        self._worker_finished_event_loop = loop
                        signal_to_connect.connect(loop.quit) # type: ignore
                        
                        # Set a timeout to prevent indefinite waiting
                        QTimer.singleShot(10000, loop.quit) # Force quit loop after 10 seconds

                        write_debug_log(f"Waiting for worker {worker} to finish...")
                        loop.exec() # Block until worker finishes or timeout
                        write_debug_log(f"Worker {worker} finished waiting.")
                        
                        self._worker_finished_event_loop = None # Cleanup

                if not thread.wait(100): # 短い時間だけ待機し、スレッドが完全に終了したか確認
                    write_debug_log(f"Thread {thread} did not exit gracefully, terminating.")
                    thread.terminate()
                else:
                    write_debug_log(f"Thread {thread} finished gracefully.")
                
                if worker:
                    worker.deleteLater()



        write_debug_log("Proceeding with application close.")
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        """Handles global key presses for image navigation."""
        if not self.input_line.hasFocus() and not self.add_single_tag_line.hasFocus():
             if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_W, Qt.Key.Key_K, Qt.Key.Key_A):
                 self._navigate_image_list(-1)
                 event.accept()
             elif event.key() in (Qt.Key.Key_Down, Qt.Key.Key_S, Qt.Key.Key_J, Qt.Key.Key_D):
                 self._navigate_image_list(1)
                 event.accept()
        # The Alt modifier logic can be added here if needed
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filters events from child widgets to implement custom keyboard shortcuts."""
        # Defensively check if target widgets exist before accessing them
        target_widgets: list[QObject] = []
        if hasattr(self, 'add_single_tag_line'):
            target_widgets.append(self.add_single_tag_line)
        if hasattr(self, 'add_tag_line'):
            target_widgets.append(self.add_tag_line)
        if hasattr(self, 'add_tag_line_append'):
            target_widgets.append(self.add_tag_line_append)

        if watched in target_widgets:
            if isinstance(event, QKeyEvent):
                if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                    if event.key() == Qt.Key.Key_Up:
                        self._navigate_image_list(-1)
                        return True # Event handled, do not process further
                    elif event.key() == Qt.Key.Key_Down:
                        self._navigate_image_list(1)
                        return True # Event handled
        
        return super().eventFilter(watched, event)

    def wheelEvent(self, event: QWheelEvent):
        """Handles mouse wheel events for image navigation."""
        if self.image_list.geometry().contains(event.position().toPoint()):
             super().wheelEvent(event)
             return
        
        # If cursor is elsewhere, perform global navigation.
        if event.angleDelta().y() > 0:
            self._navigate_image_list(-1)
        elif event.angleDelta().y() < 0:
            self._navigate_image_list(1)
        event.accept()
    
    @Slot()
    def select_image_item(self, item: QListWidgetItem):
        """Slot for when an item in the image list is clicked."""
        self._load_and_fit_image(item)

    @Slot(str, str)
    def _handle_folder_drop(self, folder_path: str, file_to_select: str | None = None):
        """Handles the logic for when a folder is dropped or selected."""
        self.input_line.setText(folder_path)
        self.reload_image_list(file_to_select) 
        self.reload_tags_only()
    
    @Slot()
    def browse_folder(self):
        """Opens a dialog to select an input folder."""
        dir_path = QFileDialog.getExistingDirectory(self, self.locale_manager.get_string("MainWindow", "Select_Input_Folder"), self.input_line.text())
        if dir_path:
            self.handle_folder_drop(dir_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handles drag enter events to accept valid file types."""
        if event.mimeData().hasUrls():
            path = Path(event.mimeData().urls()[0].toLocalFile())
            if path.is_dir() or path.suffix.lower() in constants.IMAGE_EXTENSIONS + ['.txt']:
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handles drop events to load folders or files."""
        path = Path(event.mimeData().urls()[0].toLocalFile())
        folder_path, file_to_select = None, None

        if path.is_dir():
            folder_path = str(path)
        elif path.suffix.lower() in constants.IMAGE_EXTENSIONS:
            folder_path, file_to_select = str(path.parent), path.name
        elif path.suffix.lower() == '.txt':
            folder_path = str(path.parent)
            for ext in constants.IMAGE_EXTENSIONS:
                img_path = path.with_suffix(ext)
                if img_path.is_file():
                    file_to_select = img_path.name
                    break
        
        if folder_path:
            self.handle_folder_drop(folder_path, file_to_select)
            event.acceptProposedAction()

    def resizeEvent(self, event: QResizeEvent):
        """Starts a timer to handle resizing after it has finished."""
        self._resize_timer.start()
        super().resizeEvent(event)

    @Slot()
    def _handle_resize_debounced(self):
        """Reloads the currently displayed image to fit the new window size."""
        current_item = self.image_list.currentItem()
        if current_item and self.image_label.pixmap():
            self._load_and_fit_image(current_item)
        self.update_all_button_alignments()

    def update_button_text_alignment(self, button: QPushButton):
        font_metrics = button.fontMetrics()
        text_width = font_metrics.horizontalAdvance(button.text())
        
        button_text_space = button.contentsRect().width()

        if text_width > button_text_space:
            button.setStyleSheet("QPushButton { text-align: left; padding-left: 5px; }")
        else:
            button.setStyleSheet("QPushButton { text-align: center; }")

    def update_all_button_alignments(self):
        for button in self.tag_buttons:
            if button.isVisible():
                self.update_button_text_alignment(button)
        for button in self.tag_buttons_for_image:
            if button.isVisible():
                self.update_button_text_alignment(button)


    def _navigate_image_list(self, delta: int):
        """Navigates the image list up or down by a given delta."""
        current = self.image_list.currentRow()
        new_row = current + delta
        if 0 <= new_row < self.image_list.count():
            item = self.image_list.item(new_row)
            self.image_list.setCurrentItem(item)
            self._load_and_fit_image(item)

    def _change_tag_page(self, delta: int):
        """Changes the displayed page for bulk tags."""
        new_page = self._current_page + delta
        total_pages = (len(self._all_tags) + constants.TAGS_PER_PAGE - 1) // constants.TAGS_PER_PAGE
        if 0 <= new_page < total_pages:
            self._current_page = new_page
            self.display_current_tag_page()

    def _change_image_tag_page(self, delta: int):
        """Changes the displayed page for single image tags."""
        tags_per_page = max(1, self.settings.window.tag_display_cols * self.settings.window.tag_display_rows)
        new_page = self._current_image_tag_page + delta
        total_pages = (len(self._current_image_tags) + tags_per_page - 1) // tags_per_page
        if 0 <= new_page < total_pages:
            self._current_image_tag_page = new_page
            self._display_image_tag_page()

    @Slot()
    def _add_single_tag(self):
        """Adds a new tag to the currently selected image's tag file."""
        current_item = self.image_list.currentItem()
        if not current_item:
            return

        tags_raw = self.add_single_tag_line.text().strip()
        if not tags_raw:
            return

        # (Normalization logic for full-width chars, spaces, commas...)
        tags_processed = ' '.join(tags_raw.replace('ã€€', ' ').split())
        while ',,' in tags_processed:
            tags_processed = tags_processed.replace(',,', ',')
        new_tags = [t.strip() for t in tags_processed.split(',') if t.strip()]
        if not new_tags:
            return

        image_path = Path(self.settings.paths.input_dir) / current_item.data(Qt.ItemDataRole.UserRole + 1)
        txt_path = image_path.with_suffix('.txt')

        try:
            existing_tags = []
            if txt_path.is_file():
                existing_tags = [t.strip() for t in txt_path.read_text('utf-8').split(',') if t.strip()]
            
            for tag in new_tags:
                if tag not in existing_tags:
                    existing_tags.append(tag)
            
            txt_path.write_text(', '.join(existing_tags), 'utf-8')
            self.update_log(self.locale_manager.get_string("MainWindow", "Tags_Added_To_File", txt_path_name=txt_path.name), "green")
            self.add_single_tag_line.clear()
            self._load_image_tags(image_path)
            self.reload_tags_only()
        except Exception as e:
            self.update_log(self.locale_manager.get_string("MainWindow", "Error_Adding_Tags", txt_path_name=txt_path.name, e=e), "red")

    def _delete_image_tag(self, tag_to_delete: str):
        """Deletes a tag from the currently selected image's tag file."""
        write_debug_log(f"[_delete_image_tag] Attempting to delete tag: {tag_to_delete}")
        current_item = self.image_list.currentItem()
        if not current_item:
            write_debug_log("[_delete_image_tag] No image item selected.")
            return

        image_path = Path(self.settings.paths.input_dir) / current_item.data(Qt.ItemDataRole.UserRole + 1)
        txt_path = image_path.with_suffix('.txt')
        
        write_debug_log(f"[_delete_image_tag] Image path: {image_path}, TXT path: {txt_path}")

        if not txt_path.is_file():
            write_debug_log(f"[_delete_image_tag] TXT file not found: {txt_path}")
            return
        
        try:
            tags = [t.strip() for t in txt_path.read_text('utf-8').split(',') if t.strip()]
            write_debug_log(f"[_delete_image_tag] Existing tags before deletion: {tags}")

            if tag_to_delete in tags:
                tags.remove(tag_to_delete)
                write_debug_log(f"[_delete_image_tag] Tags after removal: {tags}")
                txt_path.write_text(', '.join(tags), 'utf-8')
                write_debug_log(f"[_delete_image_tag] Successfully wrote tags to file: {txt_path.name}")
                self.update_log(self.locale_manager.get_string("MainWindow", "Tag_Deleted_From_File", tag_name=tag_to_delete, file_name=txt_path.name), "green")
                self._load_image_tags(image_path)
                write_debug_log("[_delete_image_tag] UI updated after tag deletion.")
            else:
                write_debug_log(f"[_delete_image_tag] Tag '{tag_to_delete}' not found in file: {txt_path}")
        except Exception as e:
            write_debug_log(f"[_delete_image_tag] Error during tag deletion: {e}")
            self.update_log(self.locale_manager.get_string("MainWindow", "Error_Deleting_Tag", tag_name=tag_to_delete, file_name=txt_path.name, e=e), "red")

    def add_tag_all(self, prepend: bool):
        if self._is_bulk_deleting:
            return
        tags_to_add = (self.add_tag_line if prepend else self.add_tag_line_append).text().strip()
        if not tags_to_add:
            return
        
        if QMessageBox.question(self, self.locale_manager.get_string("MainWindow", "Bulk_Add_Confirmation"), self.locale_manager.get_string("MainWindow", "Confirm_Bulk_Add_Tag", tags_to_add=tags_to_add)) == QMessageBox.StandardButton.Yes:
            self._start_bulk_tag_worker('add', input_dir=Path(self.settings.paths.input_dir), tags=tags_to_add, prepend=prepend)
    
    def delete_tag_all(self, tag_to_delete: str):
        """Starts a bulk process to delete a tag from all .txt files."""
        if self._is_bulk_deleting:
            return
        if QMessageBox.question(self, self.locale_manager.get_string("MainWindow", "Bulk_Delete_Confirmation"), self.locale_manager.get_string("MainWindow", "Confirm_Bulk_Delete_Tag", tag_to_delete=tag_to_delete)) == QMessageBox.StandardButton.Yes:
            self._start_bulk_tag_worker('delete', input_dir=Path(self.settings.paths.input_dir), tag=tag_to_delete)

    # --- Thread and Process Management ---

    def toggle_download_or_start_tagging(self):
        """Main action button logic: starts or stops download/tagging."""
        self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Process_Generic"), "black")
        QApplication.processEvents() # Ensure the log message is displayed immediately
        if self._is_downloading:
            self._stop_download_thread()
        elif self._tagger_thread and self._tagger_thread.isRunning():
            self._stop_tagging_thread()
        else:
            if self._is_model_available():
                self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Tagging_Process"), "black")
                self._start_tagging_thread()
            else:
                self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Model_Download"), "black")
                self._start_download_thread()

    def _start_tagging_thread(self):
        """Initializes and starts the TaggerThreadWorker."""
        if self._tagger_thread and self._tagger_thread.isRunning():
            return
        
        self._always_overwrite = False
        self._always_skip = False
        
        self._update_ui_for_processing(True, 'tagging')

        # Get selected file to prioritize it
        selected_path: Path | None = None
        current_item = self.image_list.currentItem()
        if current_item:
            relative_path = current_item.data(Qt.ItemDataRole.UserRole + 1)
            selected_path = Path(self.settings.paths.input_dir) / relative_path

        self._tagger_thread = QThread()
        self._tagger_worker = TaggerThreadWorker(self.settings, self._show_overwrite_dialog, self.locale_manager.get_string, selected_file_path=selected_path)
        self._tagger_worker.moveToThread(self._tagger_thread)
        
        self._tagger_worker.log_message.connect(self.update_log)
        self._tagger_worker.model_status_changed.connect(self._check_model_status_and_update_ui)
        self._tagger_worker.finished.connect(self._on_tagger_finished)
        self._tagger_thread.started.connect(self._tagger_worker.run_tagging)
        
        self._tagger_thread.start()

    def _stop_tagging_thread(self):
        """Requests the tagging thread to stop."""
        if self._tagger_worker:
            self.update_log(self.locale_manager.get_string("MainWindow", "Stopping_Tagging_Process"), "orange")
            self.run_button.setText(self.locale_manager.get_string("Constants", "Stopping_Process"))
            self.run_button.setEnabled(False)
            self._tagger_worker.stop()
            QApplication.processEvents() # Ensure UI remains responsive during stop

    def _start_download_thread(self):
        """Initializes and starts the DownloaderWorker."""
        if self._download_thread and self._download_thread.isRunning():
            return

        self._is_downloading = True
        self._update_ui_for_processing(True, 'downloading')
        self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Model_Download"), "black")

        self._download_thread = QThread()
        self._downloader_worker = DownloaderWorker(self.locale_manager.get_string)
        self._downloader_worker.moveToThread(self._download_thread)

        self._downloader_worker.log_message.connect(self.update_log)
        self._downloader_worker.progress_update.connect(self._update_download_progress)
        self._downloader_worker.download_finished.connect(self._on_download_finished)
        self._download_thread.started.connect(self._downloader_worker.run_download)
        
        self._download_thread.start()

    def _stop_download_thread(self):
        """Requests the download thread to stop."""
        if self._downloader_worker:
            self.update_log(self.locale_manager.get_string("MainWindow", "Stopping_Model_Download"), "orange")
            self.run_button.setText(self.locale_manager.get_string("Constants", "Stopping_Process"))
            self.run_button.setEnabled(False)
            self._downloader_worker.stop()
    
    def _start_bulk_tag_worker(self, mode: str, **kwargs: Any):
        """Initializes and starts the BulkTagWorker for add/delete operations."""
        self._is_bulk_deleting = True
        self._set_bulk_controls_enabled(False)
        self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Bulk_Tag_Operation", mode=mode), "blue")

        self._bulk_tag_thread = QThread()
        worker = BulkTagWorker(self.locale_manager.get_string)
        self._bulk_tag_worker = worker
        worker.moveToThread(self._bulk_tag_thread)

        worker.log_message.connect(self.update_log)
        worker.finished.connect(self._on_bulk_tag_finished)
        
        if mode == 'add':
            self._bulk_tag_thread.started.connect(lambda: worker.run_bulk_add(kwargs['input_dir'], kwargs['tags'], kwargs['prepend']))
        else: # delete
            self._bulk_tag_thread.started.connect(lambda: worker.run_bulk_delete(kwargs['input_dir'], kwargs['tag']))

        self._bulk_tag_thread.start()

    # --- Thread Finished Slots ---

    @Slot()
    def _on_tagger_finished(self):
        """Cleans up after the tagging thread has finished."""
        self.reload_tags_only()
        # Reload settings from disk as the worker may have updated the verification status
        config = load_config()
        self.settings = load_settings(config)
        self._update_ui_for_processing(False, 'tagging')
        if self.image_list.count() > 0:
            self._load_and_fit_image(self.image_list.item(0))
        
        if self._tagger_thread:
            self._tagger_thread.quit()
            self._tagger_thread.wait()
            if self._tagger_worker:
                self._tagger_worker.deleteLater()
            self._tagger_thread.deleteLater()
            self._tagger_thread = self._tagger_worker = None

    @Slot(bool)
    def _on_download_finished(self, success: bool):
        """Cleans up after the download thread has finished."""
        write_debug_log(f"Download finished signal received with success: {success}")
        self._is_downloading = False
        self._set_main_controls_enabled(True) # Re-enable main controls

        if success:
            self.update_log(self.locale_manager.get_string("MainWindow", "Model_Download_Complete"), "green")
            # Reload settings from disk as the worker may have updated the verification status
            config = load_config()
            self.settings = load_settings(config)
            self._check_model_status_and_update_ui() # On success, check status to show "TAG" button
        else:
            self.update_log(self.locale_manager.get_string("MainWindow", "Model_Download_Failed"), "red")
            self._check_model_status_and_update_ui(force_download=True) # On failure/stop, force "Download" button
            
        if self._download_thread:
            self._download_thread.quit()
            self._download_thread.wait()
            if self._downloader_worker:
                self._downloader_worker.deleteLater()
            self._download_thread.deleteLater()
            self._download_thread = self._downloader_worker = None

    @Slot()
    def _on_tag_loader_finished(self):
        """Cleans up after the tag loader thread has finished."""
        if self.tag_thread:
            self.tag_thread.quit()
            self.tag_thread.wait()
            if self.tag_worker:
                self.tag_worker.deleteLater()
            self.tag_thread.deleteLater()
            self.tag_thread = self.tag_worker = None

    @Slot()
    def _on_bulk_tag_finished(self):
        """Cleans up after the bulk tag worker thread has finished."""
        self.reload_tags_only() # This will re-enable controls on finish
        if self._bulk_tag_thread:
            self._bulk_tag_thread.quit()
            self._bulk_tag_thread.wait()
            if self._bulk_tag_worker:
                self._bulk_tag_worker.deleteLater()
            self._bulk_tag_thread.deleteLater()
            self._bulk_tag_thread = self._bulk_tag_worker = None

   

    def save_current_config(self):
        """Updates the settings object with the current UI state and saves it to file."""
        write_debug_log(self.locale_manager.get_string("MainWindow", "Saving_UI_Settings"))
        
        # Only update geometry if in main view and not maximized/minimized
        if self.central_widget.currentWidget() == self.main_view_widget and not self.isMaximized() and not self.isMinimized():
            self.settings.window.geometry = f"{self.width()}x{self.height()}+{self.x()}+{self.y()}"
        
        save_config(self.settings)
        self.update_log(self.locale_manager.get_string("MainWindow", "Settings_Saved"), "green")
        
    def _is_model_available(self) -> bool:
        """Checks if the model is verified and essential files exist."""
        if not self.settings.model.verified:
            return False
        if not constants.MODEL_PATH.is_file():
            return False
        if not constants.TAGS_CSV_PATH.is_file():
            return False
        return True

    def _get_log_color_map(self) -> dict[str, str]:
        """Returns the appropriate color map based on the detected theme."""
        theme_colors = {
            True: {"red": constants.COLOR_LOG_ERROR_DARK, "green": constants.COLOR_LOG_SUCCESS_DARK, "blue": constants.COLOR_LOG_INFO_DARK, "orange": constants.COLOR_LOG_WARN_DARK, "black": constants.COLOR_LOG_DEFAULT_DARK},
            False: {"red": constants.COLOR_LOG_ERROR_LIGHT, "green": constants.COLOR_LOG_SUCCESS_LIGHT, "blue": constants.COLOR_LOG_INFO_LIGHT, "orange": constants.COLOR_LOG_WARN_LIGHT, "black": constants.COLOR_LOG_DEFAULT_LIGHT}
        }
        return theme_colors[self._is_dark_theme]

    def create_slider_group(self, layout: QGridLayout, section: str, min_val: float, max_val: float, step: float, keys: Mapping[str, int]):
        """Creates a labeled slider and connects it to the settings object."""
        resolution = 1 / step if step < 1 else 1
        
        for key, row_index in keys.items():
            is_float = step < 1
            label_suffix = self.locale_manager.get_string("MainWindow", "Threshold_Suffix") if section == 'Thresholds' else self.locale_manager.get_string("MainWindow", "Max_Count_Suffix")
            label = QLabel(f"{key.capitalize()} {label_suffix}:")
            
            settings_section = getattr(self.settings, section.lower())
            initial_val = getattr(settings_section, key)
                
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(min_val * resolution), int(max_val * resolution))
            slider.setValue(int(initial_val * resolution))
            
            value_label = QLabel(f"{initial_val:.2f}" if is_float else str(int(initial_val)))
            value_label.setFixedWidth(50)

            def update_value(value: int, k: str = key, s: Any = settings_section, res: float = resolution, v_label: QLabel = value_label, is_flt: bool = is_float):
                real_val = value / res
                v_label.setText(f"{real_val:.2f}" if is_flt else str(int(real_val)))
                setattr(s, k, real_val if is_flt else int(real_val))

            slider.valueChanged.connect(update_value)
            
            layout.addWidget(label, row_index, 0)
            layout.addWidget(slider, row_index, 1)
            layout.addWidget(value_label, row_index, 2)

    @Slot(str, str)
    def update_log(self, message: str, color: str = "black"):
        """Appends a colored, timestamped message to the log output."""
        html_color = self._log_color_map.get(color, self._log_color_map["black"])
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        html_message = f'<span style="color:{html_color};">{timestamp}{message}</span>'
        
        self.log_output.append(html_message)

        if self.log_output.document().blockCount() > constants.MAX_LOG_LINES:
            cursor = self.log_output.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.NextBlock, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _check_model_status_and_update_ui(self, auto_start_download: bool = False, force_download: bool = False):
        """Checks for model files and updates the run button's state and appearance."""
        if not force_download and self._is_model_available():
            self.run_button.setText(self.locale_manager.get_string("Constants", "Tag_Button_Text"))
            self.run_button.setStyleSheet(constants.STYLE_BTN_GREEN)
            self.run_button.setEnabled(True)
        else:
            self.run_button.setText(self.locale_manager.get_string("Constants", "Download_Start_No_Model"))
            self.run_button.setStyleSheet(constants.STYLE_BTN_ORANGE)
            self.run_button.setEnabled(True)
            if auto_start_download:
                self.update_log(self.locale_manager.get_string("MainWindow", "Info_Model_NotFound_Start_Download"), "orange")

    @Slot(int, float, float)
    def _update_download_progress(self, percentage: int, downloaded_mb: float, total_mb: float):
        """Updates the run button text with the download percentage."""
        if self._is_downloading:
            stop_text = self.locale_manager.get_string("Constants", "Stop_Button_Text")
            button_text = f"{stop_text} ({percentage}%) {downloaded_mb:.1f}/{total_mb:.1f} MB"
            self.run_button.setText(button_text)
    
    def _animate_loading_label(self):
        """Animates the loading label with dots."""
        self.loading_state = (self.loading_state + 1) % 4
        dots = '.' * self.loading_state
        self.loading_label.setText(f"{self.locale_manager.get_string('Constants', 'Loading_Tag_List')}{dots}")


    @Slot()
    def show_enlarged_image(self):
        """Shows the currently selected image, positioned next to the tag panel."""
        if not self._original_image_pixmap or self._original_image_pixmap.isNull():
            return

        screen_geom = QApplication.primaryScreen().availableGeometry()
        
        tag_panel_widget = self.tag_display_grid.parentWidget().parentWidget() # Get the splitter child widget
        if not tag_panel_widget:
            return
            
        global_top_left = tag_panel_widget.mapToGlobal(QPoint(0,0))
        tag_panel_global_rect = QRect(global_top_left, tag_panel_widget.size())
        
        available_width = tag_panel_global_rect.left()
        available_height = screen_geom.height()

        if self._original_image_pixmap.height() == 0:
            return
        img_ratio = self._original_image_pixmap.width() / self._original_image_pixmap.height()
        
        dialog_width = available_width
        dialog_height = int(dialog_width / img_ratio) if img_ratio > 0 else available_height

        if dialog_height > available_height:
            dialog_height = available_height
            dialog_width = int(dialog_height * img_ratio)
        
        dialog_width = max(200, dialog_width)
        dialog_height = max(200, dialog_height)

        # --- MODIFIED: Position dialog to the left of the tag panel ---
        dialog_x = tag_panel_global_rect.left() - dialog_width
        dialog_y = screen_geom.y() + (screen_geom.height() - dialog_height) // 2
        
        if self._image_viewer_dialog is None:
            # Launch in navigation mode (default)
            self._image_viewer_dialog = ImageViewerDialog(self) # Removed tag_panel_global_rect
            self._image_viewer_dialog.finished.connect(self._image_viewer_dialog_closed)
            self._image_viewer_dialog.nextImageRequested.connect(lambda: self._navigate_image_list(1))
            self._image_viewer_dialog.prevImageRequested.connect(lambda: self._navigate_image_list(-1))

        self._image_viewer_dialog.show_image(self._original_image_pixmap, dialog_width, dialog_height)
        self._image_viewer_dialog.setGeometry(dialog_x, dialog_y, dialog_width, dialog_height)
        self._image_viewer_dialog.show()
        self._image_viewer_dialog.activateWindow()

    @Slot()
    def _image_viewer_dialog_closed(self):
        """Slot to clean up the reference to the image viewer dialog when it closes."""
        self._image_viewer_dialog = None

    @Slot()
    def _show_grid_view(self):
        """Switches the central widget to the GridViewWidget."""
        image_paths = self._get_image_paths(Path(self.settings.paths.input_dir))
        if not image_paths:
            QMessageBox.information(self, self.locale_manager.get_string("MainWindow", "No_Images_Found_Title"), self.locale_manager.get_string("MainWindow", "No_Images_Found_Message"))
            return

        self.central_widget.setCurrentWidget(self.grid_view_widget)
        self.setWindowTitle(f"{constants.MSG_WINDOW_TITLE} - Grid View")
        self.grid_view_widget.load_images(image_paths)
        self.showMaximized()
        self.update_log(self.locale_manager.get_string("MainWindow", "Switched_To_Grid_View"), "blue")

    @Slot()
    def _show_main_view(self):
        """Switches the central widget back to the main view."""
        self.central_widget.setCurrentWidget(self.main_view_widget)
        self.setWindowTitle(constants.MSG_WINDOW_TITLE)
        self.showNormal() # Or restore previous geometry
        self.update_log(self.locale_manager.get_string("MainWindow", "Switched_Back_To_Main_View"), "blue")
        self.reload_tags_only()
        self._load_and_fit_image(self.image_list.currentItem())

    def _show_overwrite_dialog(self, file_path: Path) -> bool:
        """
        Called from a worker thread to display an overwrite confirmation dialog in the GUI thread.
        It safely blocks the worker thread and returns the user's response (True/False) using a signal and QEventLoop.
        """
        # This method is called from a non-GUI thread.
        # We need to wait for the result from the GUI thread.
        
        self._overwrite_response = None
        loop = QEventLoop()
        self._overwrite_event_loop = loop

        # Emit a signal to the main thread to show the dialog
        self.overwrite_request.emit(file_path)

        # Wait until the main thread signals that it's done.
        loop.exec()

        response = self._overwrite_response
        
        # Clean up
        self._overwrite_event_loop = None
        self._overwrite_response = None
        
        return response if response is not None else False

    @Slot(Path)
    def _handle_overwrite_request(self, file_path: Path):
        """This slot runs in the GUI thread."""
        # It shows the dialog and sets the response.
        response = self._ask_overwrite_confirmation(file_path)
        self._overwrite_response = response
        
        # Quit the event loop to unblock the worker thread.
        if self._overwrite_event_loop:
            self._overwrite_event_loop.quit()

    def _ask_overwrite_confirmation(self, file_path: Path) -> bool:
        """Method that actually displays the dialog and asks for confirmation from the user. Executed in the GUI thread."""
        if self._always_overwrite:
            return True
        if self._always_skip:
            return False

    
        msg = QMessageBox()
        msg.setWindowTitle(self.locale_manager.get_string("MainWindow", "Overwrite_Confirmation_Title"))
        msg.setText(self.locale_manager.get_string("MainWindow", "Overwrite_Confirmation_Message", path_name=file_path.name))
        msg.setIcon(QMessageBox.Icon.Question)
        
        btn_yes = msg.addButton(self.locale_manager.get_string("MainWindow", "Overwrite"), QMessageBox.ButtonRole.YesRole)
        _ = msg.addButton(self.locale_manager.get_string("MainWindow", "Skip"), QMessageBox.ButtonRole.NoRole)
        btn_yes_all = msg.addButton(self.locale_manager.get_string("MainWindow", "Always_Overwrite"), QMessageBox.ButtonRole.YesRole)
        btn_no_all = msg.addButton(self.locale_manager.get_string("MainWindow", "Always_Skip"), QMessageBox.ButtonRole.NoRole)
        
        msg.exec()
        clicked_button = msg.clickedButton()

        if clicked_button == btn_yes_all:
            self._always_overwrite = True
        elif clicked_button == btn_no_all:
            self._always_skip = True
        
        if clicked_button is None: # type: ignore
            return False

        return clicked_button in (btn_yes, btn_yes_all)

    @Slot(str)
    def _update_input_dir(self, text: str):
        """Updates the input directory path in the settings."""
        self.settings.paths.input_dir = text