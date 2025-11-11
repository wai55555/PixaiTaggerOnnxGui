import configparser
from typing import Any, Callable, TypeVar
from dataclasses import dataclass, is_dataclass, fields
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

@dataclass
class Thresholds:
    general: float
    character: float

@dataclass
class Limits:
    general: int
    character: int

@dataclass
class Behavior:
    enable_solo_character_limit: bool
    convert_underscore_to_space: bool

@dataclass
class Window:
    geometry: str
    tag_display_rows: int = 6
    tag_display_cols: int = 5

@dataclass
class Model:
    verified: bool

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
    debug: Debug
    language_code: str

# --- Config Utilities (from config_utils.py) ---
from constants import BASE_DIR, CONFIG_PATH, MODEL_DIR_NAME

def get_default_config() -> configparser.ConfigParser:
    """Returns a ConfigParser object with default settings."""
    config = configparser.ConfigParser()
    DEFAULT_CONFIG = {
        'Paths': {'input_dir': str(BASE_DIR / "inputs"), 'model_dir': MODEL_DIR_NAME, 'model_filename': 'model.onnx'},
        'Thresholds': {'general': '0.40', 'character': '0.65'},
        'Limits': {'general': '55', 'character': '1'},
        'Behavior': {'enable_solo_character_limit': 'True', 'convert_underscore_to_space': 'True'},
        'Window': {'geometry': '986x976+50+50', 'tag_display_rows': '6', 'tag_display_cols': '5'},
        'Model': {'verified': 'False'},
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
            character=config.getfloat('Thresholds', 'character', fallback=0.65)
        ),
        limits=Limits(
            general=config.getint('Limits', 'general', fallback=55),
            character=config.getint('Limits', 'character', fallback=1)
        ),
        behavior=Behavior(
            enable_solo_character_limit=config.getboolean('Behavior', 'enable_solo_character_limit', fallback=True),
            convert_underscore_to_space=config.getboolean('Behavior', 'convert_underscore_to_space', fallback=True)
        ),
        window=Window(
            geometry=config.get('Window', 'geometry', fallback='986x976+50+50'),
            tag_display_rows=config.getint('Window', 'tag_display_rows', fallback=6),
            tag_display_cols=config.getint('Window', 'tag_display_cols', fallback=5)
        ),
        model=Model(
            verified=config.getboolean('Model', 'verified', fallback=False)
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
def update_model_verification_status(is_verified: bool, get_string: GetString):
    """
    Loads config, sets model verification status, and saves it.
    Used by worker threads to update model status.
    """
    try:
        config = load_config()
        settings = load_settings(config)
        if settings.model.verified != is_verified:
            settings.model.verified = is_verified
            save_config(settings)
            if is_verified:
                write_debug_log(get_string("ConfigUtils", "ModelVerified_Success_Debug"), get_string)
            else:
                write_debug_log(get_string("ConfigUtils", "ModelUnverified_Debug"), get_string)
    except Exception as e:
        write_debug_log(get_string("ConfigUtils", "ModelVerification_Update_Failed_Debug", e=e), get_string)
