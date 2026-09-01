import configparser
from pathlib import Path
from typing import Any

from utils import write_debug_log


class LocaleManager:
    def __init__(self, lang_code: str, base_dir: Path, *fallback_dirs: Path | None):
        self.lang_code = lang_code
        self.base_dir = base_dir
        # frozen ビルドでは同梱 .ini が _internal/lang に入り、base_dir（exe隣）とは
        # 別になる。通常は起動時に exe 隣へ配置されるが、書き込めなかった場合に備えて
        # リソース側も探索する。
        self.search_dirs: list[Path] = [base_dir]
        for extra in fallback_dirs:
            if extra is not None and extra not in self.search_dirs:
                self.search_dirs.append(extra)
        self.translations = self._load_translations()

    def _find_ini(self, file_name: str) -> Path | None:
        for directory in self.search_dirs:
            candidate = directory / file_name
            if candidate.is_file():
                return candidate
        return None

    def _load_translations(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        fallback_path = self._find_ini("en.ini")
        primary_path = self._find_ini(f"{self.lang_code}.ini")

        # Load English first as a per-key fallback, then overlay the selected language so
        # any key a translation file is missing (e.g. a whole new [CaptionCore] section)
        # still resolves to English instead of showing the raw key.
        paths = [fallback_path]
        if self.lang_code != "en" and primary_path != fallback_path:
            paths.append(primary_path)
        for path in paths:
            if path is None:
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
