import sys
from pathlib import Path
from typing import Mapping

# --- Global constant ---
BASE_DIR = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.ini"
LOG_FILE_PATH = BASE_DIR / "debug_log.txt"

# --- Model-related constants ---
MODEL_SIZE_BYTES = 1271365853
MODEL_SHA256_HASH = "b444747c34b22c7a52e1855e94b1509a25b2a096c466fa854a59117070529d2b"
MODEL_DIR_NAME = "pixai-tagger-v0.9-onnx"
MODEL_PATH = BASE_DIR / MODEL_DIR_NAME / "model.onnx"
MODEL_POINTER_PATH = BASE_DIR / MODEL_DIR_NAME / "model_pointer.txt"
TAGS_CSV_PATH = BASE_DIR / MODEL_DIR_NAME / "selected_tags.csv"
DOWNLOAD_URLS: Mapping[Path, str] = {
    MODEL_PATH: "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/resolve/main/model.onnx",
    MODEL_POINTER_PATH: "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/raw/main/model.onnx",
    TAGS_CSV_PATH: "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/resolve/main/selected_tags.csv",
}

# --- Application settings ---
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp']
TAGS_PER_PAGE = 16
TAGS_PER_PAGE_FOR_IMAGE = 20
MAX_LOG_LINES = 1000

# --- UI TEXT ---
MSG_WINDOW_TITLE = "PixAI Tagger 0.9 onnx GUI (Viewer/Bulk Edit)"

# --- Style Sheet Colors ---
STYLE_BTN_GREEN = "QPushButton { font-size: 16pt; padding: 10px; background-color: #4CAF50; color: white; }"
STYLE_BTN_BLUE = "QPushButton { font-size: 16pt; padding: 10px; background-color: #2196F3; color: white; }"
STYLE_BTN_ORANGE = "QPushButton { font-size: 16pt; padding: 10px; background-color: #FF9800; color: white; }"
STYLE_BTN_RED = "QPushButton { font-size: 16pt; padding: 10px; background-color: #F44336; color: white; }"

# Light Theme Colors (current colors)
COLOR_LOG_SUCCESS_LIGHT = "#00AA00"
COLOR_LOG_ERROR_LIGHT = "#FF0000"
COLOR_LOG_INFO_LIGHT = "#0000FF"
COLOR_LOG_WARN_LIGHT = "#FF8C00"
COLOR_LOG_DEFAULT_LIGHT = "#000000"

# Dark Theme Colors (adjusted for dark background)
COLOR_LOG_SUCCESS_DARK = "#90EE90" # Light green
COLOR_LOG_ERROR_DARK = "#FF6347"  # Tomato
COLOR_LOG_INFO_DARK = "#ADD8E6"   # Light blue
COLOR_LOG_WARN_DARK = "#FFD700"   # Gold
COLOR_LOG_DEFAULT_DARK = "#FFFFFF" # White
