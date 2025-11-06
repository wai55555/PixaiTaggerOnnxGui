import sys
from pathlib import Path
from typing import Any, Callable
from datetime import datetime
import hashlib
import configparser

# Get the base directory whether this module is compiled into an executable or not.
BASE_DIR = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
LOG_FILE_PATH = BASE_DIR / "debug_log.txt"
CONFIG_PATH = BASE_DIR / "config.ini"

# Global variable to cache the logging state
_DEBUG_LOG_ENABLED = None

def _is_debug_log_enabled() -> bool:
    """Checks config.ini to see if debug logging is enabled. Caches the result."""
    global _DEBUG_LOG_ENABLED
    if _DEBUG_LOG_ENABLED is not None:
        return _DEBUG_LOG_ENABLED

    try:
        if not CONFIG_PATH.is_file():
            _DEBUG_LOG_ENABLED = False # Default to disabled if no config exists
            return False

        config = configparser.ConfigParser()
        config.read(CONFIG_PATH, encoding='utf-8')
        _DEBUG_LOG_ENABLED = config.getboolean('Debug', 'debug_log', fallback=False)
        return _DEBUG_LOG_ENABLED
    except Exception:
        _DEBUG_LOG_ENABLED = False # Default to disabled on any error
        return False

def nowtag() -> str:
    """Return the current time as a string in the format [YYYY-MM-DD HH:MM:SS]."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")

def write_debug_log(message: str, get_string: Callable[[str, str, Any], str] | None = None):
    if not _is_debug_log_enabled():
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
        _get_string = get_string if get_string else lambda section, key, **kwargs: key
        print(f"{_get_string('Utils', 'Log_Write_Failed', message=message)}", file=sys.stderr)

def log_dbg(msg: str, get_string: Callable[[str, str, Any], str] | None = None):
    write_debug_log(msg, get_string)

def calculate_sha256(file_path: Path, chunk_size=8192) -> str:
    """Calculate the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (FileNotFoundError, OSError):
        return ""
