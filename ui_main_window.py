from typing import TYPE_CHECKING

from PySide6.QtCore import (
    Qt
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QTextEdit, QSizePolicy, QSplitter, QStackedWidget
)
from PySide6.QtGui import (
    QIcon
)


import constants

from custom_widgets import PathLineEdit, TagListWidget
from custom_dialogs import ClickableLabel
from grid_view_widget import GridViewWidget

if TYPE_CHECKING:
    from main_window import MainWindow

class Ui_MainWindow(object):
    def setup_ui(self, main_window: 'MainWindow'):
        main_window.setWindowTitle(constants.MSG_WINDOW_TITLE)
        main_window.setWindowIcon(QIcon(str(constants.BASE_DIR / "icons/app_icon.ico")))
        try:
            if main_window.settings.window and main_window.settings.window.geometry:
                geometry_str = main_window.settings.window.geometry
                geom = geometry_str.split('+')
                size = geom[0].split('x')
                main_window.resize(int(size[0]), int(size[1]))
                main_window.move(int(geom[1]), int(geom[2]))
            else:
                raise ValueError("Geometry is not set")
        except (ValueError, IndexError):
            main_window.resize(950, 720)
            main_window.move(50, 50)
        
        main_window.setAcceptDrops(True)

        # Central stacked widget for view switching
        main_window.central_widget = QStackedWidget(main_window)
        main_window.setCentralWidget(main_window.central_widget)

        # Main View Setup
        main_window.main_view_widget = self._create_main_view(main_window)
        main_window.central_widget.addWidget(main_window.main_view_widget)

        # Grid View Setup
        main_window.grid_view_widget = GridViewWidget(main_window.settings, main_window.locale_manager)
        main_window.central_widget.addWidget(main_window.grid_view_widget)
        
        self._connect_signals(main_window)
        main_window._check_model_status_and_update_ui(auto_start_download=True)  # type: ignore

    def _connect_signals(self, main_window: 'MainWindow'):
        """Connects all signals to their corresponding slots."""
        main_window.image_list.itemClicked.connect(main_window.select_image_item)
        main_window.input_line.editingFinished.connect(main_window.reload_image_list)
        main_window.input_line.textChanged.connect(main_window._update_input_dir)  # type: ignore
        main_window.input_line.folder_dropped.connect(main_window._handle_folder_drop)  # type: ignore
        main_window.run_button.clicked.connect(main_window.toggle_download_or_start_tagging)
        main_window.grid_view_widget.back_to_main_requested.connect(main_window._show_main_view)  # type: ignore
        main_window._resize_timer.timeout.connect(main_window._handle_resize_debounced)  # type: ignore
        main_window.overwrite_dialog_requested.connect(main_window._handle_overwrite_request)

    def _create_main_view(self, main_window: 'MainWindow') -> QWidget:
        """Constructs the main view widget with its layout and components."""
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        main_window.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_panel = self._create_left_panel(main_window)
        right_panel = self._create_right_panel(main_window)

        main_window.splitter.addWidget(left_panel)
        main_window.splitter.addWidget(right_panel)
        main_window.splitter.setStretchFactor(0, 1)
        main_window.splitter.setStretchFactor(1, 3)
        
        layout.addWidget(main_window.splitter)
        return main_widget

    def _create_left_panel(self, main_window: 'MainWindow') -> QWidget:
        """Creates the left panel containing the image list."""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        main_window.image_list = TagListWidget(get_string=main_window.locale_manager.get_string)
        main_window.image_list.setMaximumWidth(500)
        layout.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Image_File_List")))
        layout.addWidget(main_window.image_list)
        return left_widget

    def _create_right_panel(self, main_window: 'MainWindow') -> QWidget:
        """Creates the right panel with all the controls and viewers."""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)

        main_window.right_vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        
        input_group = self._create_input_group(main_window)
        viewer_group = self._create_viewer_group(main_window)
        bulk_actions_group = self._create_bulk_actions_group(main_window)
        log_group = self._create_log_group(main_window)

        layout.addWidget(input_group)
        main_window.right_vertical_splitter.addWidget(viewer_group)
        main_window.right_vertical_splitter.addWidget(bulk_actions_group)
        main_window.right_vertical_splitter.addWidget(log_group)
        
        main_window.right_vertical_splitter.setSizes([400, 200, 100])
        main_window.right_vertical_splitter.setStretchFactor(0, 4)
        main_window.right_vertical_splitter.setStretchFactor(1, 2)
        main_window.right_vertical_splitter.setStretchFactor(2, 1)

        layout.addWidget(main_window.right_vertical_splitter)
        return right_widget

    def _create_input_group(self, main_window: 'MainWindow') -> QGroupBox:
        """Creates the input folder selection group box."""
        group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Input_Folder"), main_window)
        layout = QVBoxLayout(group)
        controls_layout = QHBoxLayout()
        
        main_window.input_line = PathLineEdit(main_window, get_string=main_window.locale_manager.get_string)
        main_window.input_line.setText(main_window.settings.paths.input_dir)
        main_window.input_line.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Drag_Drop_Folder_Placeholder"))
        
        browse_button = QPushButton(main_window.locale_manager.get_string("MainWindow", "Browse_Button"))
        browse_button.clicked.connect(main_window.browse_folder)
        
        main_window.grid_view_button = QPushButton("3x3 edit")
        main_window.grid_view_button.setToolTip(main_window.locale_manager.get_string("MainWindow", "Switch_To_Grid_View"))
        main_window.grid_view_button.clicked.connect(main_window._show_grid_view)  # type: ignore
        
        controls_layout.addWidget(main_window.input_line)
        controls_layout.addWidget(browse_button)
        controls_layout.addWidget(main_window.grid_view_button)
        layout.addLayout(controls_layout)
        return group

    def _create_viewer_group(self, main_window: 'MainWindow') -> QSplitter:
        """Creates the splitter for the image viewer and tag editor."""
        img_tag_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        main_window.image_label = ClickableLabel()
        main_window.image_label.doubleClicked.connect(main_window.show_enlarged_image)
        main_window.image_label.setMinimumSize(225, 225)
        main_window.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_window.image_label.setToolTip(main_window.locale_manager.get_string("MainWindow", "ImageViewer_Tooltip"))
        
        tag_panel_widget = self._create_single_image_tag_panel(main_window)
        
        img_tag_splitter.addWidget(main_window.image_label)
        img_tag_splitter.addWidget(tag_panel_widget)
        img_tag_splitter.setSizes([200, 300])
        img_tag_splitter.setStretchFactor(0, 1)
        img_tag_splitter.setStretchFactor(1, 1)
        return img_tag_splitter

    def _create_single_image_tag_panel(self, main_window: 'MainWindow') -> QWidget:
        """Creates the panel for viewing and editing tags of a single image."""
        tag_panel_widget = QWidget()
        tag_panel = QVBoxLayout(tag_panel_widget)
        tag_panel.setContentsMargins(0, 0, 0, 0)
        tag_panel.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Image_Tags_Label")))
        
        tag_grid_container = QWidget()
        main_window.tag_display_grid = QGridLayout(tag_grid_container)
        tag_grid_container.setLayout(main_window.tag_display_grid)
        min_grid_height = main_window.settings.window.tag_display_rows * (main_window._tag_button_min_height + 5)  # type: ignore
        tag_grid_container.setMinimumHeight(min_grid_height)
        tag_grid_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        tag_panel.addWidget(tag_grid_container)
        
        page_nav_layout = QHBoxLayout()
        main_window.image_tag_prev_page_btn = QPushButton(main_window.locale_manager.get_string("MainWindow", "Previous_Page"))
        main_window.image_tag_prev_page_btn.clicked.connect(lambda: main_window._change_image_tag_page(-1))  # type: ignore
        main_window.image_tag_next_page_btn = QPushButton(main_window.locale_manager.get_string("MainWindow", "Next_Page"))
        main_window.image_tag_next_page_btn.clicked.connect(lambda: main_window._change_image_tag_page(1))  # type: ignore
        page_nav_layout.addWidget(main_window.image_tag_prev_page_btn)
        page_nav_layout.addWidget(main_window.image_tag_next_page_btn)
        tag_panel.addLayout(page_nav_layout)
        
        tag_panel.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Add_Single_Tag_Label")))
        add_tag_layout = QHBoxLayout()
        main_window.add_single_tag_line = QLineEdit()
        main_window.add_single_tag_line.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Tags_Comma_Separated_Placeholder"))
        main_window.add_single_tag_line.setToolTip(main_window.locale_manager.get_string("MainWindow", "AddTag_Hover_Tooltip"))
        main_window.add_single_tag_line.returnPressed.connect(main_window._add_single_tag)  # type: ignore
        main_window.add_single_tag_line.installEventFilter(main_window)
        main_window.add_single_tag_button = QPushButton(main_window.locale_manager.get_string("MainWindow", "Add_Button"))
        main_window.add_single_tag_button.clicked.connect(main_window._add_single_tag)  # type: ignore
        add_tag_layout.addWidget(main_window.add_single_tag_line)
        add_tag_layout.addWidget(main_window.add_single_tag_button)
        tag_panel.addLayout(add_tag_layout)
        
        tag_panel.addStretch(1)
        return tag_panel_widget

    def _create_bulk_actions_group(self, main_window: 'MainWindow') -> QWidget:
        """Creates the widget containing all bulk action and settings controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        bulk_edit_layout = QHBoxLayout()
        bulk_delete_group = self._create_bulk_delete_group(main_window)
        bulk_add_group = self._create_bulk_add_group(main_window)
        bulk_edit_layout.addWidget(bulk_delete_group, 3)
        bulk_edit_layout.addWidget(bulk_add_group, 1)
        layout.addLayout(bulk_edit_layout)

        settings_group = self._create_settings_group(main_window)
        layout.addWidget(settings_group)
        
        main_window.run_button = QPushButton(main_window.locale_manager.get_string("Constants", "Tag_Button_Text"))
        main_window.run_button.setStyleSheet(constants.STYLE_BTN_GREEN)
        layout.addWidget(main_window.run_button)
        
        return widget

    def _create_bulk_delete_group(self, main_window: 'MainWindow') -> QGroupBox:
        """Creates the group for bulk tag deletion."""
        group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Bulk_Delete_Tags"))
        layout = QVBoxLayout(group)
        
        main_window.loading_label = QLabel(main_window.locale_manager.get_string("Constants", "Loading_Tag_List"))
        main_window.loading_label.setStyleSheet("font-weight:bold;")
        layout.addWidget(main_window.loading_label)
        
        main_window.tag_button_grid = QGridLayout()
        for i in range(4):
            main_window.tag_button_grid.setColumnStretch(i, 1)
        layout.addLayout(main_window.tag_button_grid)
        
        page_nav_layout = QHBoxLayout()
        main_window.prev_page_btn = QPushButton(main_window.locale_manager.get_string("MainWindow", "Previous_16_Items"))
        main_window.prev_page_btn.clicked.connect(lambda: main_window._change_tag_page(-1))  # type: ignore
        main_window.next_page_btn = QPushButton(main_window.locale_manager.get_string("MainWindow", "Next_16_Items"))
        main_window.next_page_btn.clicked.connect(lambda: main_window._change_tag_page(1))  # type: ignore
        page_nav_layout.addWidget(main_window.prev_page_btn)
        page_nav_layout.addWidget(main_window.next_page_btn)
        layout.addLayout(page_nav_layout)
        
        return group

    def _create_bulk_add_group(self, main_window: 'MainWindow') -> QGroupBox:
        """Creates the group for bulk tag addition."""
        group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Bulk_Add_Tags"))
        layout = QVBoxLayout(group)
        
        layout.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Files")))
        main_window.add_tag_line = QLineEdit()
        main_window.add_tag_line.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Placeholder"))
        main_window.add_tag_line.setToolTip(main_window.locale_manager.get_string("MainWindow", "Comma_Recommended_Tooltip"))
        main_window.add_tag_button = QPushButton(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Txt_Files"))
        main_window.add_tag_button.setStyleSheet(constants.STYLE_BTN_BLUE)
        main_window.add_tag_button.clicked.connect(lambda: main_window.add_tag_all(prepend=True))
        main_window.add_tag_line.installEventFilter(main_window)
        layout.addWidget(main_window.add_tag_line)
        layout.addWidget(main_window.add_tag_button)

        layout.addWidget(QLabel(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Files_Append")))
        main_window.add_tag_line_append = QLineEdit()
        main_window.add_tag_line_append.setPlaceholderText(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Placeholder"))
        main_window.add_tag_line_append.setToolTip(main_window.locale_manager.get_string("MainWindow", "Comma_Recommended_Tooltip"))
        main_window.add_tag_button_append = QPushButton(main_window.locale_manager.get_string("MainWindow", "Add_Tags_To_All_Txt_Files_Append"))
        main_window.add_tag_button_append.setStyleSheet(constants.STYLE_BTN_BLUE)
        main_window.add_tag_button_append.clicked.connect(lambda: main_window.add_tag_all(prepend=False))
        main_window.add_tag_line_append.installEventFilter(main_window)
        layout.addWidget(main_window.add_tag_line_append)
        layout.addWidget(main_window.add_tag_button_append)

        layout.addStretch(1)
        return group

    def _create_settings_group(self, main_window: 'MainWindow') -> QWidget:
        """Creates the widget for threshold and limit sliders."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        thresh_group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Tag_Threshold"))
        thresh_layout = QGridLayout(thresh_group)
        main_window.create_slider_group(thresh_layout, 'Thresholds', 0.0, 1.0, 0.01, {'general': 0, 'character': 1})
        
        limit_group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Max_Tags"))
        limit_layout = QGridLayout(limit_group)
        main_window.create_slider_group(limit_layout, 'Limits', 1, 150, 1, {'general': 0})
        main_window.create_slider_group(limit_layout, 'Limits', 1, 10, 1, {'character': 1})
        
        layout.addWidget(thresh_group)
        layout.addWidget(limit_group)
        return widget

    def _create_log_group(self, main_window: 'MainWindow') -> QGroupBox:
        """Creates the group for the execution log."""
        group = QGroupBox(main_window.locale_manager.get_string("MainWindow", "Execution_Log"), main_window)
        layout = QVBoxLayout(group)
        main_window.log_output = QTextEdit()
        main_window.log_output.setReadOnly(True)
        main_window.log_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(main_window.log_output)
        main_window.log_output.setMinimumHeight(100)
        main_window.log_output.setMaximumHeight(400)
        return group
