from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Protocol
from datetime import datetime
import hashlib
import configparser

# Get the base directory whether this module is compiled into an executable or not.
BASE_DIR = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
LOG_FILE_PATH = BASE_DIR / "debug_log.txt"
CONFIG_PATH = BASE_DIR / "config.ini"

class GetString(Protocol):
    def __call__(self, section: str, key: str, **kwargs: Any) -> str: ...

def default_get_string_fallback(section: str, key: str, **kwargs: Any) -> str:
    """Default fallback for get_string if no localization function is provided."""
    return key

class DebugSettings:
    _instance: DebugSettings | None = None

    def __init__(self):
        self.debug_log_enabled: bool = False
        try:
            if CONFIG_PATH.is_file():
                config = configparser.ConfigParser()
                config.read(CONFIG_PATH, encoding='utf-8')
                self.debug_log_enabled = config.getboolean('Debug', 'debug_log', fallback=False)
        except Exception:
            self.debug_log_enabled = False

    @classmethod
    def get_instance(cls) -> 'DebugSettings':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_debug_settings() -> DebugSettings:
    return DebugSettings.get_instance()

def nowtag() -> str:
    """Return the current time as a string in the format [YYYY-MM-DD HH:MM:SS]."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")

def write_debug_log(message: str, get_string: GetString | None = None):
    _get_string: GetString = get_string if get_string else default_get_string_fallback # Moved initialization here
    if not get_debug_settings().debug_log_enabled:
        return
        
    if not message.strip():
        return
        
    lines = message.split('\n')
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            for line in lines:
                if line.strip():
                    f.write(nowtag() + line.strip() + "\n")
            f.flush()
    except Exception:
        print(f"{_get_string('Utils', 'Log_Write_Failed', message=message)}", file=sys.stderr)

def log_dbg(msg: str, get_string: GetString | None = None):
    write_debug_log(msg, get_string)

def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (FileNotFoundError, OSError):
        return ""
