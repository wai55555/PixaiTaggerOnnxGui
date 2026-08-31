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
        fallback_path = self.base_dir / "en.ini"
        primary_path = self.base_dir / f"{self.lang_code}.ini"

        # Load English first as a per-key fallback, then overlay the selected language so
        # any key a translation file is missing (e.g. a whole new [CaptionCore] section)
        # still resolves to English instead of showing the raw key.
        paths = [fallback_path]
        if self.lang_code != "en" and primary_path != fallback_path:
            paths.append(primary_path)
        for path in paths:
            if not path.is_file():
                continue
            try:
                config.read(path, encoding="utf-8")
            except Exception as e:
                write_debug_log(f"Failed to read language file {path}: {e}")
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
