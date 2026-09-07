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

    def _load_file_layered(self, config: configparser.ConfigParser, file_name: str) -> None:
        """`file_name` を search_dirs の優先度が低い順（末尾から）に読み込む。

        configparser.read() は後で読んだファイルのキーが先に読んだものを上書きするので、
        末尾（同梱リソース側）から読んで先頭（exe隣の書き込み可能ディレクトリ）を最後に
        読めば、exe隣のファイルが優先されつつ、そこに無いキーだけ同梱版の値へ「ファイル
        単位」ではなく「キー単位」でフォールバックする。

        exe隣の lang/*.ini は一度作られたら二度と上書きされない（models/ と同じ「既存
        ファイルは触らない」方針、ユーザーの手編集を保護するため）。ファイル単位で最初に
        見つかった1つだけを読む実装だと、アップデートで追加された翻訳キーが exe隣の古い
        ファイルには無いまま埋まらず raw key 表示になっていた（PR#16 レビュー指摘）。
        """
        for directory in reversed(self.search_dirs):
            path = directory / file_name
            if not path.is_file():
                continue
            try:
                config.read(path, encoding="utf-8")
            except Exception as e:
                write_debug_log(f"Failed to read language file {path}: {e}")

    def _load_translations(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()

        # Load English first as a per-key fallback, then overlay the selected language so
        # any key a translation file is missing (e.g. a whole new [CaptionCore] section)
        # still resolves to English instead of showing the raw key.
        self._load_file_layered(config, "en.ini")
        if self.lang_code != "en":
            self._load_file_layered(config, f"{self.lang_code}.ini")
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
