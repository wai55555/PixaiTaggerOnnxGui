from pathlib import Path
import requests
import threading
from typing import Callable
from collections import Counter

from PySide6.QtCore import QObject, Signal, Slot

from utils import write_debug_log, calculate_sha256, GetString, default_get_string_fallback
from constants import (
    DOWNLOAD_URLS, MODEL_PATH, TAGS_CSV_PATH, MODEL_POINTER_PATH
)
from app_settings import AppSettings, update_model_verification_status
from get_pointer_huggingface import get_model_info_from_pointer
from tagging_core import setup_tagger_from_settings, process_image_loop, get_image_paths_recursive

from pathlib import Path
import requests
import threading
from typing import Callable
from collections import Counter

from PySide6.QtCore import QObject, Signal, Slot

from utils import write_debug_log, calculate_sha256, GetString, default_get_string_fallback
from constants import (
    DOWNLOAD_URLS, MODEL_PATH, TAGS_CSV_PATH, MODEL_POINTER_PATH
)
from app_settings import AppSettings, update_model_verification_status
from get_pointer_huggingface import get_model_info_from_pointer
from tagging_core import setup_tagger_from_settings, process_image_loop, get_image_paths_recursive

class DownloaderWorker(QObject):
    """Downloads model files and verifies their integrity."""
    log_message = Signal(str, str)
    progress_update = Signal(int, float, float) # percentage, downloaded_mb, total_mb
    download_finished = Signal(bool) # success/failure

    def __init__(self, get_string: GetString | None = None):
        super().__init__()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self._stop_event = threading.Event()
        self._file_sizes: dict[Path, int] = {} # To store expected file sizes

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def _download_single_file(self, file_path: Path, url: str, expected_sha256: str | None = None) -> bool:
        """
        Downloads a single file with progress updates and optional SHA256 verification.
        Returns True on success, False otherwise.
        """
        if self.is_stopped():
            return False

        file_name = file_path.name
        total_size = self._file_sizes.get(file_path, 0)
        
        if file_path.exists():
            if file_path == TAGS_CSV_PATH:
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV", file_name=file_path.name), "blue")
                write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV_Debug", file_path_name=file_path.name)), self.get_string)
                return True

            local_size = file_path.stat().st_size
            if total_size > 0:
                if local_size > total_size:
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_LocalFileTooLarge", file_name=file_path.name), "red")
                    return False
                elif local_size == total_size:
                    if file_path == MODEL_PATH and expected_sha256:
                        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_VerifyingHash", file_name=file_path.name), "blue")
                        local_sha256 = calculate_sha256(file_path)
                        if local_sha256.lower() == expected_sha256.lower():
                            self._mark_model_as_verified()
                            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_HashMatch", file_name=file_path.name), "green")
                            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_HashMatch_Log", file_path_name=file_path.name)), self.get_string)
                            return True
                        else:
                            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_HashMismatch", file_name=file_path.name), "red")
                            file_path.unlink(missing_ok=True)
                            return False
                    else:
                        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Skip_Existing", file_path_name=file_path.name)), self.get_string)
                        return True

        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Downloading_Model", file_name=file_name, total_size=f"{total_size/1024/1024:.2f}"), "blue")
        
        downloaded_size = file_path.stat().st_size if file_path.exists() else 0
        mode = 'ab' if downloaded_size > 0 else 'wb'

        try:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_URL_Connect_Start", url=url)), self.get_string)
            headers = {'Range': f'bytes={downloaded_size}-'}
            response = requests.get(url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()
            content_length = int(response.headers.get('content-length', 0))
            
            if content_length > 0:
                if response.status_code == 206:
                    total_size = downloaded_size + content_length
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Resume_Download", content_length=f"{content_length/1024/1024:.2f}", total_size=f"{total_size/1024/1024:.2f}"), "black")
                else:
                    if downloaded_size > 0:
                        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Warning_200_OK_Restart"), "orange")
                        downloaded_size = 0
                        mode = 'wb'
                    total_size = content_length
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Info_FileSize_Header", total_size=f"{total_size/1024/1024:.2f}"), "black")
            
            last_percent = int(downloaded_size * 100 / total_size) if total_size > 0 else 0
            self.progress_update.emit(last_percent, downloaded_size / 1024 / 1024, total_size / 1024 / 1024)
            
            with open(file_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.is_stopped(): break
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0: 
                        percent = min(100, int(downloaded_size * 100 / total_size))
                        self.progress_update.emit(percent, downloaded_size / 1024 / 1024, total_size / 1024 / 1024)
                        last_percent = percent
            
            if self.is_stopped():
                write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Download_Aborted", file_name=file_name)), self.get_string)
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Download_Aborted_User", file_name=file_name), "red")
                return False
            
            final_size = file_path.stat().st_size
            if final_size != total_size:
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Size_Mismatch", downloaded_size=f"{final_size/1024/1024:.2f}", total_size=f"{total_size/1024/1024:.2f}"), "red")
                return False

            if file_path == MODEL_PATH and expected_sha256:
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_VerifyingHash", file_name=file_path.name), "blue")
                local_sha256 = calculate_sha256(file_path)
                if local_sha256.lower() != expected_sha256.lower():
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_HashMismatch", file_name=file_path.name), "red")
                    file_path.unlink(missing_ok=True)
                    return False
                else:
                    self._mark_model_as_verified()
            
            if not self.is_stopped():
                self.progress_update.emit(100, total_size / 1024 / 1024, total_size / 1024 / 1024)
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Download_Complete", file_name=file_name), "green")
            
            return True
                
        except requests.exceptions.RequestException as e:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Network_Error", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Network_Failed"), "red")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Details", type_e_name=type(e).__name__, e=e), "red")
            return False
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Unexpected_Error", e=e)), self.get_string)
            error_msg = self.get_string("Workers", "DownloaderWorker_Error_Unexpected_File_Access")
            self.log_message.emit(error_msg, "red")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Details", type_e_name=type(e).__name__, e=e), "red")
            return False

    def _mark_model_as_verified(self):
        try:
            update_model_verification_status(True, self.get_string)
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_ModelVerified_Success"), "green")
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_ModelVerified_Success_Debug")), self.get_string)

        except Exception as e:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_ModelVerified_Fail"), "red")
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_ModelVerified_Fail_Debug", e=e)), self.get_string)

    @Slot()
    def run_download(self):
        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Start")), self.get_string)
        all_success = True

        model_pointer_url = DOWNLOAD_URLS.get(MODEL_POINTER_PATH)
        if not model_pointer_url:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_NoPointerURL"), "red")
            self.download_finished.emit(False)
            return

        expected_sha256, expected_size = get_model_info_from_pointer(model_pointer_url, self.get_string)
        if not expected_sha256 or not expected_size:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_FailedToGetModelInfo"), "red")
            self.download_finished.emit(False)
            return
        
        self._file_sizes[MODEL_PATH] = expected_size

        for file_path, url in DOWNLOAD_URLS.items():
            if file_path == MODEL_POINTER_PATH:
                continue

            if self.is_stopped():
                all_success = False
                break

            success = self._download_single_file(file_path, url, expected_sha256 if file_path == MODEL_PATH else None)
            if not success:
                all_success = False
                break
        
        self.download_finished.emit(all_success)
        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Download_Thread_Exit")), self.get_string)

class TaggerThreadWorker(QObject):
    """Tagging Worker"""
    log_message = Signal(str, str)
    model_status_changed = Signal()
    finished = Signal()
    running_state_changed = Signal(bool)
    reload_image_list_signal = Signal()
    def __init__(self, settings: AppSettings, overwrite_checker: Callable[[Path], bool], get_string: GetString | None = None, selected_file_path: Path | None = None):
        super().__init__()
        self._settings: AppSettings = settings
        self._overwrite_checker = overwrite_checker
        self._selected_file_path = selected_file_path
        self._stop_event = threading.Event()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
    
    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self) -> bool:
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def _mark_model_as_unverified(self):
        try:
            update_model_verification_status(False, self.get_string)
            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_ModelUnverified"), "orange")
            self.model_status_changed.emit()
        except Exception as e:
            write_debug_log(str(f"Debug: Failed to save model unverified status: {e}"), self.get_string)

    @Slot()
    def run_tagging(self):
        write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Start")), self.get_string)
        self.running_state_changed.emit(True)
        write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Tagging_Process_Start")), self.get_string)
        
        try:
            tagger, settings_dict = setup_tagger_from_settings(self._settings, self.get_string)
            if not tagger or not settings_dict:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Error_Tagger_Init_Failed"), "red")
                self._mark_model_as_unverified()
                self.running_state_changed.emit(False)
                self.finished.emit()
                return
            
            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Loading_Model"), "black")

            input_dir = Path(settings_dict['INPUT_DIR'])
            image_paths = get_image_paths_recursive(input_dir)

            if self._selected_file_path and self._selected_file_path in image_paths:
                image_paths.remove(self._selected_file_path)
                image_paths.insert(0, self._selected_file_path)

            if not image_paths:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Warning_No_Image_Files", input_dir=input_dir), "orange")
                self.running_state_changed.emit(False)
                self.finished.emit()
                return

            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Total_Image_Files", count=len(image_paths)), "blue")

            def log_to_gui(message: str, color: str):
                write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Core_Log", message=message)), self.get_string)
                self.log_message.emit(message, color)

            process_image_loop(
                tagger=tagger,
                image_paths=image_paths,
                settings=settings_dict,
                overwrite_checker=self._overwrite_checker,
                log_gui=log_to_gui,
                stop_checker=self.is_stopped,
                get_string=self.get_string
            )
            
        except Exception as e:
            import traceback
            error_message = self.get_string("Workers", "TaggerThreadWorker_Fatal_Exception", type_e_name=type(e).__name__, e=e, traceback_exc=traceback.format_exc())
            self.log_message.emit(error_message, "red")
            write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Runtime_Exception", e=e, traceback_exc=traceback.format_exc())), self.get_string)
        
        finally:
            self.running_state_changed.emit(False)
            self.reload_image_list_signal.emit()
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Thread_Exit")), self.get_string)

class TagLoader(QObject):
    """Worker to asynchronously load tag files from an image folder"""
    log_message = Signal(str, str)
    tags_loaded = Signal(list)
    finished = Signal()
    def __init__(self, folder: Path, get_string: GetString | None = None):
        super().__init__()
        self.folder = folder
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self._stop_event = threading.Event()

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def run(self):
        write_debug_log(str(self.get_string("Workers", "TagLoader_Start", folder=self.folder)), self.get_string)
        counter: Counter[str] = Counter()
        files = list(self.folder.rglob("*.txt"))
        try:
            for txt in files:
                if self.is_stopped():
                    break
                try:
                    with open(txt, "r", encoding="utf-8") as f:
                        tags = [t.strip() for t in f.read().split(",") if t.strip()]
                        counter.update(tags)
                except Exception as e:
                    write_debug_log(str(self.get_string("Workers", "TagLoader_TXT_Load_Failed", txt_name=txt.name, e=e)), self.get_string)
            
            if not self.is_stopped():
                all_tags: list[tuple[str, int]] = counter.most_common() 
                self.tags_loaded.emit(all_tags)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "TagLoader_Fatal_Error", e=e)), self.get_string)
            if not self.is_stopped():
                self.tags_loaded.emit([])
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "TagLoader_Thread_Exit")), self.get_string)

class BulkTagWorker(QObject):
    """Worker to execute bulk tag editing (add/delete)"""
    log_message = Signal(str, str)
    finished = Signal()

    def __init__(self, get_string: GetString | None = None):
        super().__init__()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self._stop_event = threading.Event()

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def _process_tag_file(self, txt_file_path: Path, tag_operation_callback: Callable[[list[str]], list[str]]) -> bool:
        try:
            with open(txt_file_path, "r", encoding="utf-8") as f:
                existing_tags = [t.strip() for t in f.read().split(',') if t.strip()]
            
            modified_tags = tag_operation_callback(existing_tags)
            
            if set(existing_tags) != set(modified_tags):
                new_content = ", ".join(modified_tags)
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_File_Processing_Failed", txt_name=txt_file_path.name, e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_File_Processing_Failed", txt_name=txt_file_path.name), "red")
        return False

    @Slot(Path, str)
    def run_bulk_delete(self, input_dir: Path, tag_to_delete: str):
        write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Start", tag_to_delete=tag_to_delete)), self.get_string)
        count = 0
        try:
            for txt in input_dir.rglob("*.txt"):
                if self.is_stopped():
                    break
                
                def delete_callback(tags: list[str]) -> list[str]:
                    return [t for t in tags if t != tag_to_delete]

                if self._process_tag_file(txt, delete_callback):
                    count += 1
            
            if not self.is_stopped():
                self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Complete", count=count, tag_to_delete=tag_to_delete), "green")
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Count", count=count)), self.get_string)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Unexpected_Error_Bulk_Delete", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Unexpected_Bulk_Delete", e=e), "red")
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Thread_Exit")), self.get_string)

    @Slot(Path, str, bool)
    def run_bulk_add(self, input_dir: Path, tags_to_add: str, prepend: bool):
        write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Start", tags_to_add=tags_to_add)), self.get_string)
        count = 0
        
        new_tags_to_add = sorted(list(set([t.strip() for t in tags_to_add.split(',') if t.strip()])))
        if not new_tags_to_add:
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Warning_No_Valid_Tags_To_Add"), "orange")
            self.finished.emit()
            return

        try:
            for txt in input_dir.rglob("*.txt"):
                if self.is_stopped():
                    break
                
                def add_callback(existing_tags: list[str]) -> list[str]:
                    if prepend:
                        return [tag for tag in new_tags_to_add if tag not in existing_tags] + existing_tags
                    else:
                        return existing_tags + [tag for tag in new_tags_to_add if tag not in existing_tags]

                if self._process_tag_file(txt, add_callback):
                    count += 1

            if not self.is_stopped():
                self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Complete", count=count, tags_to_add=tags_to_add), "green")
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Count", count=count)), self.get_string)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Unexpected_Error_Bulk_Add", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Unexpected_Bulk_Add", e=e), "red")
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Thread_Exit")), self.get_string)

class TaggerThreadWorker(QObject):
    """Tagging Worker"""
    log_message = Signal(str, str)
    model_status_changed = Signal()
    finished = Signal()
    running_state_changed = Signal(bool)
    reload_image_list_signal = Signal()
    def __init__(self, settings: AppSettings, overwrite_checker: Callable[[Path], bool], get_string: GetString | None = None, selected_file_path: Path | None = None):
        super().__init__()
        self._settings: AppSettings = settings
        self._overwrite_checker = overwrite_checker
        self._selected_file_path = selected_file_path
        self._stop_event = threading.Event()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
    
    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self) -> bool:
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def _mark_model_as_unverified(self):
        try:
            update_model_verification_status(False, self.get_string)
            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_ModelUnverified"), "orange")
            self.model_status_changed.emit()
        except Exception as e:
            write_debug_log(str(f"Debug: Failed to save model unverified status: {e}"), self.get_string)

    @Slot()
    def run_tagging(self):
        write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Start")), self.get_string)
        self.running_state_changed.emit(True)
        write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Tagging_Process_Start")), self.get_string)
        
        try:
            tagger, settings_dict = setup_tagger_from_settings(self._settings, self.get_string)
            if not tagger or not settings_dict:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Error_Tagger_Init_Failed"), "red")
                self._mark_model_as_unverified()
                self.running_state_changed.emit(False)
                self.finished.emit()
                return
            
            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Loading_Model"), "black")

            input_dir = Path(settings_dict['INPUT_DIR'])
            image_paths = get_image_paths_recursive(input_dir)

            if self._selected_file_path and self._selected_file_path in image_paths:
                image_paths.remove(self._selected_file_path)
                image_paths.insert(0, self._selected_file_path)

            if not image_paths:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Warning_No_Image_Files", input_dir=input_dir), "orange")
                self.running_state_changed.emit(False)
                self.finished.emit()
                return

            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Total_Image_Files", count=len(image_paths)), "blue")

            def log_to_gui(message: str, color: str):
                write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Core_Log", message=message)), self.get_string)
                self.log_message.emit(message, color)

            process_image_loop(
                tagger=tagger,
                image_paths=image_paths,
                settings=settings_dict,
                overwrite_checker=self._overwrite_checker,
                log_gui=log_to_gui,
                stop_checker=self.is_stopped,
                get_string=self.get_string
            )
            
        except Exception as e:
            import traceback
            error_message = self.get_string("Workers", "TaggerThreadWorker_Fatal_Exception", type_e_name=type(e).__name__, e=e, traceback_exc=traceback.format_exc())
            self.log_message.emit(error_message, "red")
            write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Runtime_Exception", e=e, traceback_exc=traceback.format_exc())), self.get_string)
        
        finally:
            self.running_state_changed.emit(False)
            self.reload_image_list_signal.emit()
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Thread_Exit")), self.get_string)

class TagLoader(QObject):
    """Worker to asynchronously load tag files from an image folder"""
    log_message = Signal(str, str)
    tags_loaded = Signal(list)
    finished = Signal()
    def __init__(self, folder: Path, get_string: GetString | None = None):
        super().__init__()
        self.folder = folder
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self._stop_event = threading.Event()

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def run(self):
        write_debug_log(str(self.get_string("Workers", "TagLoader_Start", folder=self.folder)), self.get_string)
        counter: Counter[str] = Counter()
        files = list(self.folder.rglob("*.txt"))
        try:
            for txt in files:
                if self.is_stopped():
                    break
                try:
                    with open(txt, "r", encoding="utf-8") as f:
                        tags = [t.strip() for t in f.read().split(",") if t.strip()]
                        counter.update(tags)
                except Exception as e:
                    write_debug_log(str(self.get_string("Workers", "TagLoader_TXT_Load_Failed", txt_name=txt.name, e=e)), self.get_string)
            
            if not self.is_stopped():
                all_tags: list[tuple[str, int]] = counter.most_common() 
                self.tags_loaded.emit(all_tags)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "TagLoader_Fatal_Error", e=e)), self.get_string)
            if not self.is_stopped():
                self.tags_loaded.emit([])
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "TagLoader_Thread_Exit")), self.get_string)

class BulkTagWorker(QObject):
    """Worker to execute bulk tag editing (add/delete)"""
    log_message = Signal(str, str)
    finished = Signal()

    def __init__(self, get_string: GetString | None = None):
        super().__init__()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self._stop_event = threading.Event()

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def _process_tag_file(self, txt_file_path: Path, tag_operation_callback: Callable[[list[str]], list[str]]) -> bool:
        try:
            with open(txt_file_path, "r", encoding="utf-8") as f:
                existing_tags = [t.strip() for t in f.read().split(',') if t.strip()]
            
            modified_tags = tag_operation_callback(existing_tags)
            
            if set(existing_tags) != set(modified_tags):
                new_content = ", ".join(modified_tags)
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_File_Processing_Failed", txt_name=txt_file_path.name, e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_File_Processing_Failed", txt_name=txt_file_path.name), "red")
        return False

    @Slot(Path, str)
    def run_bulk_delete(self, input_dir: Path, tag_to_delete: str):
        write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Start", tag_to_delete=tag_to_delete)), self.get_string)
        count = 0
        try:
            for txt in input_dir.rglob("*.txt"):
                if self.is_stopped():
                    break
                
                def delete_callback(tags: list[str]) -> list[str]:
                    return [t for t in tags if t != tag_to_delete]

                if self._process_tag_file(txt, delete_callback):
                    count += 1
            
            if not self.is_stopped():
                self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Complete", count=count, tag_to_delete=tag_to_delete), "green")
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Count", count=count)), self.get_string)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Unexpected_Error_Bulk_Delete", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Unexpected_Bulk_Delete", e=e), "red")
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Thread_Exit")), self.get_string)

    @Slot(Path, str, bool)
    def run_bulk_add(self, input_dir: Path, tags_to_add: str, prepend: bool):
        write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Start", tags_to_add=tags_to_add)), self.get_string)
        count = 0
        
        new_tags_to_add = sorted(list(set([t.strip() for t in tags_to_add.split(',') if t.strip()])))
        if not new_tags_to_add:
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Warning_No_Valid_Tags_To_Add"), "orange")
            self.finished.emit()
            return

        try:
            for txt in input_dir.rglob("*.txt"):
                if self.is_stopped():
                    break
                
                def add_callback(existing_tags: list[str]) -> list[str]:
                    if prepend:
                        return [tag for tag in new_tags_to_add if tag not in existing_tags] + existing_tags
                    else:
                        return existing_tags + [tag for tag in new_tags_to_add if tag not in existing_tags]

                if self._process_tag_file(txt, add_callback):
                    count += 1

            if not self.is_stopped():
                self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Complete", count=count, tags_to_add=tags_to_add), "green")
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Count", count=count)), self.get_string)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Unexpected_Error_Bulk_Add", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Unexpected_Bulk_Add", e=e), "red")
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Thread_Exit")), self.get_string)