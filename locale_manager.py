import configparser
from pathlib import Path
from typing import Any

from utils import write_debug_log


class LocaleManager:
    def __init__(self, lang_code: str, base_dir: Path):
        self.lang_code = lang_code
        self.base_dir = base_dir
        self.translations = self._load_translations()

    def _load_translations(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        primary_path = self.base_dir / "lang" / f"{self.lang_code}.ini"
        fallback_path = self.base_dir / "lang" / "en.ini"

        # Try to read the primary language file
        if primary_path.is_file():
            try:
                config.read(primary_path, encoding="utf-8")
                return config
            except Exception as e:
                write_debug_log(f"Failed to read primary language file {primary_path}: {e}")

        # If primary fails or doesn't exist, fall back to English
        if fallback_path.is_file():
            try:
                config.read(fallback_path, encoding="utf-8")
            except Exception as e:
                write_debug_log(f"Failed to read fallback language file {fallback_path}: {e}")
        
        # If both fail, return an empty config to prevent crashes
        return config

    def get_string(self, section: str, key: str, **kwargs: Any) -> str:
        try:
            raw_string = self.translations.get(section, key, fallback=key)
            try:
                return raw_string.format(**kwargs)
            except (KeyError, ValueError) as e:
                # Log the formatting error but return the raw string to avoid crashing
                write_debug_log(f"LocaleManager format error for key '{key}' in section '{section}': {e}. Kwargs: {kwargs}")
                return raw_string
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Fallback to key if not found
            return key.replace("_", " ").capitalize()
