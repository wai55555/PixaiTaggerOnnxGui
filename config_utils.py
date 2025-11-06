import configparser

from utils import write_debug_log
from constants import BASE_DIR, CONFIG_PATH, MODEL_DIR_NAME
from settings_model import AppSettings, Paths, Thresholds, Limits, Behavior, Window, Model, Debug


_get_string = lambda section, key, **kwargs: key # Temporary definition for early checks

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
        'Debug': {'debug_log': 'False'}
    }
    config.read_dict(DEFAULT_CONFIG)
    return config

def load_config() -> configparser.ConfigParser:
    """Loads the config.ini file, creating it from defaults if it doesn't exist."""
    config = get_default_config()
    if CONFIG_PATH.is_file():
        config.read(CONFIG_PATH, encoding='utf-8')
        write_debug_log(_get_string("ConfigUtils", "Config_File_Load_Success", CONFIG_PATH=CONFIG_PATH))
    else:
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                config.write(f)
            write_debug_log(_get_string("ConfigUtils", "Config_File_NotFound_Create_Default", CONFIG_PATH=CONFIG_PATH))
        except Exception as e:
            write_debug_log(_get_string("ConfigUtils", "Config_File_Creation_Failed", e=e))
    return config

def load_settings(config: configparser.ConfigParser) -> AppSettings:
    """Loads settings from a ConfigParser object into an AppSettings dataclass."""
    if not config.has_section('Model'):
        config.add_section('Model')
        config.set('Model', 'verified', 'False')
    if not config.has_section('Debug'):
        config.add_section('Debug')
        config.set('Debug', 'debug_log', 'True')
        
    return AppSettings(
        paths=Paths(
            input_dir=config.get('Paths', 'input_dir'),
            model_dir=config.get('Paths', 'model_dir'),
            model_filename=config.get('Paths', 'model_filename')
        ),
        thresholds=Thresholds(
            general=config.getfloat('Thresholds', 'general'),
            character=config.getfloat('Thresholds', 'character')
        ),
        limits=Limits(
            general=config.getint('Limits', 'general'),
            character=config.getint('Limits', 'character')
        ),
        behavior=Behavior(
            enable_solo_character_limit=config.getboolean('Behavior', 'enable_solo_character_limit'),
            convert_underscore_to_space=config.getboolean('Behavior', 'convert_underscore_to_space')
        ),
        window=Window(
            geometry=config.get('Window', 'geometry'),
            tag_display_rows=config.getint('Window', 'tag_display_rows', fallback=6),
            tag_display_cols=config.getint('Window', 'tag_display_cols', fallback=5)
        ),
        model=Model(
            verified=config.getboolean('Model', 'verified', fallback=False)
        ),
        debug=Debug(
            debug_log=config.getboolean('Debug', 'debug_log', fallback=True)
        ),
        language_code=config.get('General', 'language_code', fallback="")
    )

def save_config(settings: AppSettings):
    """Saves the AppSettings object to the config.ini file."""
    write_debug_log(_get_string("ConfigUtils", "Settings_Save_Start"))
    config = configparser.ConfigParser()
    config.add_section('Paths')
    config.set('Paths', 'input_dir', settings.paths.input_dir)
    config.set('Paths', 'model_dir', settings.paths.model_dir)
    config.set('Paths', 'model_filename', settings.paths.model_filename)

    config.add_section('Thresholds')
    config.set('Thresholds', 'general', f"{settings.thresholds.general:.2f}")
    config.set('Thresholds', 'character', f"{settings.thresholds.character:.2f}")

    config.add_section('Limits')
    config.set('Limits', 'general', str(settings.limits.general))
    config.set('Limits', 'character', str(settings.limits.character))

    config.add_section('Behavior')
    config.set('Behavior', 'enable_solo_character_limit', str(settings.behavior.enable_solo_character_limit))
    config.set('Behavior', 'convert_underscore_to_space', str(settings.behavior.convert_underscore_to_space))


    config.add_section('Window')
    config.set('Window', 'geometry', settings.window.geometry)
    config.set('Window', 'tag_display_rows', str(settings.window.tag_display_rows))
    config.set('Window', 'tag_display_cols', str(settings.window.tag_display_cols))

    config.add_section('Model')
    config.set('Model', 'verified', str(settings.model.verified))

    config.add_section('Debug')
    config.set('Debug', 'debug_log', str(settings.debug.debug_log))

    config.add_section('General')
    config.set('General', 'language_code', settings.language_code)

    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            config.write(f)
        write_debug_log(_get_string("ConfigUtils", "Config_File_Save_Success", CONFIG_PATH=CONFIG_PATH))
    except Exception as e:
        write_debug_log(_get_string("ConfigUtils", "Config_File_Save_Failed", e=e))
