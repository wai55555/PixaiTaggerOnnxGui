import shutil
import sys
from pathlib import Path
from typing import Mapping

def get_resource_dir() -> Path:
    """
    Determines the resource directory, handling PyInstaller's _internal folder.
    """
    if getattr(sys, "frozen", False):
        # For bundled apps, resources like icons might be in _internal
        exe_dir = Path(sys.executable).parent
        internal_dir = exe_dir / "_internal"
        return internal_dir if internal_dir.is_dir() else exe_dir
    return Path(__file__).parent.resolve()

# --- Path Constants ---
BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent.resolve()

# RESOURCE_DIR is where bundled, non-user-editable resources are located.
# This handles PyInstaller's `_internal` folder structure.
RESOURCE_DIR = get_resource_dir()

# LANG_DIR is where user-editable translation files are located.
# It should be next to the executable.
LANG_DIR = BASE_DIR / "lang"

# User-facing paths are relative to BASE_DIR
CONFIG_PATH = BASE_DIR / "config.ini"
LOG_FILE_PATH = BASE_DIR / "debug_log.txt"

# --- Model-related constants ---
MODEL_SIZE_BYTES = 1271365853
# Directory for additional (non-PixAI) tagger/captioner models, one subdirectory per model_id.
# app_settings.Paths.model_dir/model_filename remain PixAI-only legacy fields (design.md 6.9節).
# MODELS_DIR is user-writable (next to the exe) - downloaded model.onnx files land here.
MODELS_DIR = BASE_DIR / "models"
# When frozen, the hand-authored model_config.json files are bundled under RESOURCE_DIR
# (_internal/), separate from the user-writable MODELS_DIR. Non-frozen: same directory.
MODELS_RESOURCE_DIR = RESOURCE_DIR / "models"


def _seed_bundled_model_files() -> None:
    """Copy bundled per-model files into the user-visible models/ directory.

    PyInstaller puts bundled data under `_internal/`, which users are not expected to
    open - and models/ is exactly where they drop manually-downloaded model files and
    where downloads land, so the shipped model_config.json / translation CSVs have to be
    there too. Only files that do not already exist are copied, so a user's edits and
    the multi-GB downloaded model.onnx are never touched.

    No-op when not frozen (both paths resolve to the same directory).
    """
    if MODELS_RESOURCE_DIR.resolve() == MODELS_DIR.resolve():
        return
    if not MODELS_RESOURCE_DIR.is_dir():
        return
    for src in MODELS_RESOURCE_DIR.rglob("*"):
        if not src.is_file():
            continue
        dest = MODELS_DIR / src.relative_to(MODELS_RESOURCE_DIR)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


try:
    _seed_bundled_model_files()
except Exception:
    # A read-only install directory must not stop the app from starting; discover_models()
    # also scans MODELS_RESOURCE_DIR directly, so the model list still works.
    pass
# PixAI's own directory lives under MODELS_DIR like every other model (unified 2026-08-31),
# but it still has no model_config.json - it stays the hardcoded pseudo-entry in
# model_registry.py, so discover_models()'s directory scan skips it and there is no
# duplicate model_combo entry.
MODEL_DIR_NAME = "pixai-tagger-v0.9"
_NEW_PIXAI_DIR = MODELS_DIR / MODEL_DIR_NAME
_LEGACY_PIXAI_DIR = BASE_DIR / "pixai-tagger-v0.9-onnx"
if not (_NEW_PIXAI_DIR / "model.onnx").is_file() and (_LEGACY_PIXAI_DIR / "model.onnx").is_file():
    # An update replaces app code/assets in place but never touches a user's already-
    # downloaded 1.2GB model file sitting at the pre-migration path - silently moving it
    # would be an unnecessary risk (see task.md 2026-08-31 note), and a fresh redownload
    # is wasteful, so fall back to wherever the model actually is. Once found there,
    # everything derived from this directory (translation CSVs included) also comes
    # from there, so an existing install keeps working exactly as before until the user
    # deletes/redownloads it under the new path.
    _PIXAI_DIR = _LEGACY_PIXAI_DIR
else:
    _PIXAI_DIR = _NEW_PIXAI_DIR
MODEL_PATH = _PIXAI_DIR / "model.onnx"
MODEL_POINTER_PATH = _PIXAI_DIR / "model_pointer.txt"
TAGS_CSV_PATH = _PIXAI_DIR / "selected_tags.csv"
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
STYLE_LIST_ITEM_SELECTED_DARK = "QListWidget::item:selected { background-color: #1a6b9a; color: #ffffff; }"

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
