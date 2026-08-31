import configparser
from typing import Any
from dataclasses import dataclass, field, is_dataclass, fields
from pathlib import Path

from utils import write_debug_log, GetString, default_get_string_fallback

_get_string: GetString = default_get_string_fallback

def set_get_string_func(func: GetString):
    global _get_string
    _get_string = func

# --- AppSettings Models (from settings_model.py) ---
@dataclass
class Paths:
    input_dir: str
    model_dir: str
    model_filename: str

# The tag categories that can carry their own threshold / max-count. Order is the
# order rows appear in the "詳細" (per-category) dialog. "general" and "character"
# also have sliders in the main window; the rest are dialog-only.
TAG_CATEGORY_NAMES: tuple[str, ...] = (
    "general", "character", "rating", "copyright", "artist", "meta", "model", "quality", "year",
)


def parse_touched(raw: str) -> set[str]:
    """Splits a `Thresholds.touched` / `Limits.touched` comma list into a set."""
    return {c.strip() for c in raw.split(",") if c.strip()}


def add_touched(raw: str, category: str) -> str:
    """Returns `raw` with `category` added to its comma list (sorted, de-duped)."""
    marked = parse_touched(raw)
    marked.add(category)
    return ",".join(sorted(marked))

@dataclass
class Thresholds:
    general: float
    character: float
    rating: float = 0.50
    copyright: float = 0.50
    artist: float = 0.50
    meta: float = 0.50
    model: float = 0.50
    quality: float = 0.50
    year: float = 0.50
    # Comma-separated category names the user has adjusted by hand. On a model switch
    # a category NOT listed here is reset to the new model's recommended default; a
    # listed one is left as the user set it (design: 2026-08-31 user decision).
    touched: str = ""

@dataclass
class Limits:
    general: int
    character: int
    rating: int = 0
    copyright: int = 0
    artist: int = 0
    meta: int = 0
    model: int = 0
    quality: int = 0
    year: int = 0
    touched: str = ""

@dataclass
class Behavior:
    enable_solo_character_limit: bool
    convert_underscore_to_space: bool
    # 既存 .txt が存在する場合の処理方針: ASK / OVERWRITE / SKIP / APPEND。
    # 既定値により、このキーを持たない旧 config.ini でも従来どおり動作する。
    existing_file_mode: str = "ASK"

@dataclass
class Window:
    geometry: str
    tag_display_rows: int = 6
    tag_display_cols: int = 5

@dataclass
class Model:
    model_id: str = "pixai-tagger-v0.9"
    verified_models: dict[str, bool] = field(default_factory=dict)

@dataclass
class Caption:
    task: str = "MORE_DETAILED_CAPTION"

@dataclass
class Debug:
    debug_log: bool

@dataclass
class AppSettings:
    paths: Paths
    thresholds: Thresholds
    limits: Limits
    behavior: Behavior
    window: Window
    model: Model
    caption: Caption
    debug: Debug
    language_code: str

# --- Config Utilities (from config_utils.py) ---
from constants import BASE_DIR, CONFIG_PATH, MODEL_DIR_NAME

def get_default_config() -> configparser.ConfigParser:
    """Returns a ConfigParser object with default settings."""
    config = configparser.ConfigParser()
    DEFAULT_CONFIG = {
        'Paths': {'input_dir': str(BASE_DIR / "inputs"), 'model_dir': MODEL_DIR_NAME, 'model_filename': 'model.onnx'},
        'Thresholds': {'general': '0.40', 'character': '0.65', 'rating': '0.50', 'copyright': '0.50', 'artist': '0.50', 'meta': '0.50', 'model': '0.50', 'quality': '0.50', 'year': '0.50', 'touched': ''},
        'Limits': {'general': '55', 'character': '1', 'rating': '0', 'copyright': '0', 'artist': '0', 'meta': '0', 'model': '0', 'quality': '0', 'year': '0', 'touched': ''},
        'Behavior': {'enable_solo_character_limit': 'True', 'convert_underscore_to_space': 'True', 'existing_file_mode': 'ASK'},
        'Window': {'geometry': '986x976+50+50', 'tag_display_rows': '6', 'tag_display_cols': '5'},
        'Model': {'model_id': 'pixai-tagger-v0.9', 'verified_models': ''},
        'Caption': {'task': 'MORE_DETAILED_CAPTION'},
        'Debug': {'debug_log': 'False'},
        'General': {'language_code': ''}
    }
    config.read_dict(DEFAULT_CONFIG)
    return config

def load_config() -> configparser.ConfigParser:
    """Loads the config.ini file, creating it from defaults if it doesn't exist."""
    config = get_default_config()
    if CONFIG_PATH.is_file():
            
        config.read(CONFIG_PATH, encoding='utf-8')
        write_debug_log(_get_string("ConfigUtils", "Config_File_Load_Success", CONFIG_PATH=CONFIG_PATH), _get_string)
    else:
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                config.write(f)
            write_debug_log(_get_string("ConfigUtils", "Config_File_NotFound_Create_Default", CONFIG_PATH=CONFIG_PATH), _get_string)
        except Exception as e:
            write_debug_log(_get_string("ConfigUtils", "Config_File_Creation_Failed", e=e), _get_string)
    return config

def _parse_verified_models(config: configparser.ConfigParser) -> dict[str, bool]:
    """
    Parses the `[Model] verified_models` field ("model_id:True,other_id:False").

    Back-compat: a config.ini written before multi-model support has a single
    `[Model] verified` bool for the (implicit) PixAI tagger, and no `verified_models`
    key at all. Migrate it into `{"pixai-tagger-v0.9": <that bool>}` in that case.

    Note: `config.has_option('Model', 'model_id')` is NOT a reliable way to detect
    this, because `get_default_config()` (called by `load_config()` before the file
    is read) already seeds a `model_id` default into the same ConfigParser object -
    so by the time we see it here, `model_id` is present whether or not the file on
    disk ever had it. `verified`, on the other hand, is never added by the current
    defaults, so its presence really does mean "this came from the file".
    """
    if config.has_option('Model', 'verified'):
        legacy_verified = config.getboolean('Model', 'verified', fallback=False)
        return {"pixai-tagger-v0.9": legacy_verified}

    raw = config.get('Model', 'verified_models', fallback='')
    result: dict[str, bool] = {}
    for entry in raw.split(','):
        entry = entry.strip()
        if not entry or ':' not in entry:
            continue
        model_id, _, value = entry.partition(':')
        model_id = model_id.strip()
        if model_id:
            result[model_id] = value.strip().lower() == 'true'
    return result

def load_settings(config: configparser.ConfigParser) -> AppSettings:
    """Loads settings from a ConfigParser object into an AppSettings dataclass."""

    return AppSettings(
        paths=Paths(
            input_dir=config.get('Paths', 'input_dir', fallback=str(BASE_DIR / "inputs")),
            model_dir=config.get('Paths', 'model_dir', fallback=MODEL_DIR_NAME),
            model_filename=config.get('Paths', 'model_filename', fallback='model.onnx')
        ),
        thresholds=Thresholds(
            general=config.getfloat('Thresholds', 'general', fallback=0.40),
            character=config.getfloat('Thresholds', 'character', fallback=0.65),
            rating=config.getfloat('Thresholds', 'rating', fallback=0.50),
            copyright=config.getfloat('Thresholds', 'copyright', fallback=0.50),
            artist=config.getfloat('Thresholds', 'artist', fallback=0.50),
            meta=config.getfloat('Thresholds', 'meta', fallback=0.50),
            model=config.getfloat('Thresholds', 'model', fallback=0.50),
            quality=config.getfloat('Thresholds', 'quality', fallback=0.50),
            year=config.getfloat('Thresholds', 'year', fallback=0.50),
            touched=config.get('Thresholds', 'touched', fallback='')
        ),
        limits=Limits(
            general=config.getint('Limits', 'general', fallback=55),
            character=config.getint('Limits', 'character', fallback=1),
            rating=config.getint('Limits', 'rating', fallback=0),
            copyright=config.getint('Limits', 'copyright', fallback=0),
            artist=config.getint('Limits', 'artist', fallback=0),
            meta=config.getint('Limits', 'meta', fallback=0),
            model=config.getint('Limits', 'model', fallback=0),
            quality=config.getint('Limits', 'quality', fallback=0),
            year=config.getint('Limits', 'year', fallback=0),
            touched=config.get('Limits', 'touched', fallback='')
        ),
        behavior=Behavior(
            enable_solo_character_limit=config.getboolean('Behavior', 'enable_solo_character_limit', fallback=True),
            convert_underscore_to_space=config.getboolean('Behavior', 'convert_underscore_to_space', fallback=True),
            existing_file_mode=config.get('Behavior', 'existing_file_mode', fallback='ASK')
        ),
        window=Window(
            geometry=config.get('Window', 'geometry', fallback='986x976+50+50'),
            tag_display_rows=config.getint('Window', 'tag_display_rows', fallback=6),
            tag_display_cols=config.getint('Window', 'tag_display_cols', fallback=5)
        ),
        model=Model(
            model_id=config.get('Model', 'model_id', fallback='pixai-tagger-v0.9'),
            verified_models=_parse_verified_models(config)
        ),
        caption=Caption(
            task=config.get('Caption', 'task', fallback='MORE_DETAILED_CAPTION')
        ),
        debug=Debug(
            debug_log=config.getboolean('Debug', 'debug_log', fallback=False) # Default is False for debug_log
        ),
        language_code=config.get('General', 'language_code', fallback="")
    )

def save_config(settings: AppSettings):
    """Saves the AppSettings object to the config.ini file."""
    write_debug_log(_get_string("ConfigUtils", "Settings_Save_Start"), _get_string)
    config = configparser.ConfigParser()

    def _save_dataclass_to_config(config_parser: configparser.ConfigParser, dataclass_instance: Any, section_name: str):
        if not config_parser.has_section(section_name):
            config_parser.add_section(section_name)
        
        for field_info in fields(dataclass_instance):
            value = getattr(dataclass_instance, field_info.name)
            
            if is_dataclass(value):
                # Recursively handle nested dataclasses
                _save_dataclass_to_config(config_parser, value, field_info.name.capitalize())
            else:
                # Convert value to string for configparser
                if isinstance(value, float):
                    config_parser.set(section_name, field_info.name, f"{value:.2f}")
                elif isinstance(value, Path):
                    config_parser.set(section_name, field_info.name, str(value))
                elif isinstance(value, dict):
                    # e.g. Model.verified_models -> "model_id:True,other_id:False"
                    config_parser.set(section_name, field_info.name, ",".join(f"{k}:{v}" for k, v in value.items()))
                else:
                    config_parser.set(section_name, field_info.name, str(value))

    # Handle top-level fields of AppSettings
    for field_info in fields(settings):
        value = getattr(settings, field_info.name)
        if is_dataclass(value):
            _save_dataclass_to_config(config, value, field_info.name.capitalize())
        else:
            # Handle language_code which is directly in AppSettings but belongs to 'General' section
            if field_info.name == 'language_code':
                if not config.has_section('General'):
                    config.add_section('General')
                config.set('General', field_info.name, str(value))
            else:
                # This case should ideally not be hit if AppSettings only contains nested dataclasses
                # and language_code is handled specifically.
                pass # Or raise an error if unexpected direct fields exist


    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            config.write(f)
        write_debug_log(_get_string("ConfigUtils", "Config_File_Save_Success", CONFIG_PATH=CONFIG_PATH), _get_string)
    except Exception as e:
        write_debug_log(_get_string("ConfigUtils", "Config_File_Save_Failed", e=e), _get_string)
def update_model_verification_status(model_id: str, is_verified: bool, get_string: GetString):
    """
    Loads config, sets the verification status for a single model_id, and saves it.
    Used by worker threads to update model status without clobbering settings the
    GUI thread may have changed concurrently (config is reloaded fresh from disk).
    """
    try:
        config = load_config()
        settings = load_settings(config)
        if settings.model.verified_models.get(model_id) != is_verified:
            settings.model.verified_models[model_id] = is_verified
            save_config(settings)
            if is_verified:
                write_debug_log(get_string("ConfigUtils", "ModelVerified_Success_Debug"), get_string)
            else:
                write_debug_log(get_string("ConfigUtils", "ModelUnverified_Debug"), get_string)
    except Exception as e:
        write_debug_log(get_string("ConfigUtils", "ModelVerification_Update_Failed_Debug", e=e), get_string)