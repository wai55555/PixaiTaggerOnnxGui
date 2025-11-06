import configparser
from pathlib import Path
from typing import Any

class LocaleManager:
    def __init__(self, lang_code: str, base_dir: Path):
        self.lang_code = lang_code
        self.base_dir = base_dir
        self.translations = self._load_translations()

    def _load_translations(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        lang_file_path = self.base_dir / "lang" / f"{self.lang_code}.ini"
        if lang_file_path.is_file():
            config.read(lang_file_path, encoding="utf-8")
        else:
            # Fallback to English or default if language file not found
            fallback_path = self.base_dir / "lang" / "en.ini" # Assuming an English fallback
            if fallback_path.is_file():
                config.read(fallback_path, encoding="utf-8")
            # If no fallback, config will be empty
        return config

    def get_string(self, section: str, key: str, **kwargs: Any) -> str:
        try:
            raw_string = self.translations.get(section, key)
            return raw_string.format(**kwargs)
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Fallback to key if not found
            return key.replace("_", " ").capitalize().format(**kwargs)
