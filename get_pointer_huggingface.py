import sys
import locale
from pathlib import Path
from typing import Callable, Any

import requests

if __name__ == '__main__':
    # If executed directly as a script, add the project root to the path.
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from constants import MODEL_POINTER_PATH, BASE_DIR
from utils import write_debug_log
from config_utils import load_config, load_settings
from locale_manager import LocaleManager

def get_model_info_from_pointer(url: str, get_string: Callable[[str, str, Any], str] | None = None) -> tuple[str | None, int | None]:
    """
        Retrieve the pointer file from the specified URL and extract the SHA256 OID and size.

    Args:
        url (str): model pointer file
        get_string (Callable): function to retrieve localized strings.

    Returns:
        A tuple containing the SHA256 OID (str) and the size (int). (None, None) if retrieval or parsing fails.
    """
    _get_string = get_string if get_string else lambda section, key, **kwargs: key

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception if the status code is not in the 200 range.
        
        lines = response.text.strip().split('\n')
        oid_line = next((line for line in lines if line.startswith('oid sha256:')), None)
        size_line = next((line for line in lines if line.startswith('size ')), None)

        if not oid_line or not size_line:
            write_debug_log(_get_string("GetPointerHuggingface", "Error_OidOrSizeNotFound"), get_string=_get_string)
            return None, None

        sha256_oid = oid_line.split(':')[1].strip()
        size = int(size_line.split(' ')[1].strip())
        
        return sha256_oid, size

    except requests.exceptions.RequestException as e:
        write_debug_log(_get_string("GetPointerHuggingface", "Error_FetchFailed", e=e), get_string=_get_string)
        return None, None
    except (IndexError, ValueError) as e:
        write_debug_log(_get_string("GetPointerHuggingface", "Error_ParseFailed", e=e), get_string=_get_string)
        return None, None

if __name__ == '__main__':
    config = load_config()
    settings = load_settings(config)
    os_lang = locale.getdefaultlocale()[0].split('_')[0] if locale.getdefaultlocale()[0] else "en"
    locale_manager = LocaleManager(settings.language_code or os_lang, BASE_DIR)
    _ = locale_manager.get_string

    sha256, size = get_model_info_from_pointer(MODEL_POINTER_PATH, get_string=_)
    if sha256 and size:
        print(f"Model SHA256: {sha256}")
        print(f"Model Size: {size} bytes")

