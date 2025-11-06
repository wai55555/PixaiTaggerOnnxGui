import configparser
import requests
import time
from pathlib import Path
from typing import Any, Mapping, Callable
from collections import Counter

from PySide6.QtCore import QObject, Signal, Slot, QMutex

from utils import write_debug_log, calculate_sha256
from constants import (
    DOWNLOAD_URLS, MODEL_PATH, TAGS_CSV_PATH, MODEL_POINTER_PATH
)
from settings_model import AppSettings
from get_pointer_huggingface import get_model_info_from_pointer
from tagging_core import setup_tagger_from_settings, process_image_loop, get_image_paths_recursive, TagCategory

class DownloaderWorker(QObject):
    """Worker for downloading model files."""
    log_message = Signal(str, str)
    download_finished = Signal(bool)
    progress_update = Signal(int, float, float) # percent, downloaded_mb, total_mb
    
    def __init__(self, get_string: Callable[[str, str, Any], str] | None = None):
        super().__init__()
        self._mutex = QMutex()
        self._is_paused = False
        self._is_stopped = False
        self.get_string = get_string if get_string else lambda section, key, **kwargs: key
        self._file_sizes: dict[Path, int] = {}

    def stop(self):
        self._mutex.lock()
        self._is_stopped = True
        self._mutex.unlock()

    def is_stopped(self):
        self._mutex.lock()
        stopped = self._is_stopped
        self._mutex.unlock()
        return stopped

    def _mark_model_as_verified(self):
        """Loads config, sets model as verified, and saves it."""
        try:
            from config_utils import load_config, load_settings, save_config
            config = load_config()
            settings = load_settings(config)
            settings.model.verified = True
            save_config(settings)
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_ModelVerified_Success"), "green")
            write_debug_log(self.get_string("Workers", "DownloaderWorker_ModelVerified_Success_Debug"))
        except Exception as e:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_ModelVerified_Fail"), "red")
            write_debug_log(self.get_string("Workers", "DownloaderWorker_ModelVerified_Fail_Debug", e=e))

    @Slot()
    def run_download(self):
        write_debug_log(self.get_string("Workers", "DownloaderWorker_Start"))
        all_success = True

        # 1. Get model info from pointer file
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

        # Main download loop
        for file_path, url in DOWNLOAD_URLS.items():
            if file_path == MODEL_POINTER_PATH:
                continue # Skip pointer file itself

            if self.is_stopped():
                all_success = False
                break

            # File validation before download
            if file_path.exists():
                if file_path == TAGS_CSV_PATH:
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV", file_name=file_path.name), "blue")
                    write_debug_log(self.get_string("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV_Debug", file_path_name=file_path.name))
                    continue # Skip download if selected_tags.csv already exists
                
                local_size = file_path.stat().st_size
                expected_file_size = self._file_sizes.get(file_path)

                if expected_file_size:
                    if local_size > expected_file_size:
                        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_LocalFileTooLarge", file_name=file_path.name), "red")
                        local_size = 0
                    elif local_size == expected_file_size:
                        if file_path == MODEL_PATH:
                            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_VerifyingHash", file_name=file_path.name), "blue")
                            local_sha256 = calculate_sha256(file_path)
                            if local_sha256.lower() == expected_sha256.lower():
                                self._mark_model_as_verified()
                                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_HashMatch", file_name=file_path.name), "green")
                                write_debug_log(self.get_string("Workers", "DownloaderWorker_HashMatch_Log", file_path_name=file_path.name))
                                continue
                            else:
                                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_HashMismatch", file_name=file_path.name), "red")
                                all_success = False
                                break # Stop process, require user action
                        else: # For other files like CSV
                            write_debug_log(self.get_string("Workers", "DownloaderWorker_Skip_Existing", file_path_name=file_path.name))
                            continue

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_name = file_path.name
            total_size = self._file_sizes.get(file_path, 0)
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Downloading_Model", file_name=file_name, total_size=f"{total_size/1024/1024:.2f}"), "blue")
            
            downloaded_size = file_path.stat().st_size if file_path.exists() else 0
            mode = 'ab' if downloaded_size > 0 else 'wb'

            try:
                write_debug_log(self.get_string("Workers", "DownloaderWorker_URL_Connect_Start", url=url))
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
                        # Update even if the percentage hasn't changed
                        if total_size > 0: 
                            percent = min(100, int(downloaded_size * 100 / total_size))
                            self.progress_update.emit(percent, downloaded_size / 1024 / 1024, total_size / 1024 / 1024)
                            last_percent = percent
                
                if self.is_stopped():
                    write_debug_log(self.get_string("Workers", "DownloaderWorker_Download_Aborted", file_name=file_name))
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Download_Aborted_User", file_name=file_name), "red")
                    all_success = False
                    break
                
                # Final verification after download
                final_size = file_path.stat().st_size
                if final_size != total_size:
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Size_Mismatch", downloaded_size=f"{final_size/1024/1024:.2f}", total_size=f"{total_size/1024/1024:.2f}"), "red")
                    all_success = False
                    break

                if file_path == MODEL_PATH:
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_VerifyingHash", file_name=file_path.name), "blue")
                    local_sha256 = calculate_sha256(file_path)
                    if local_sha256.lower() != expected_sha256.lower():
                        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_HashMismatch", file_name=file_path.name), "red")
                        file_path.unlink()
                        all_success = False
                        break
                    else:
                        self._mark_model_as_verified()
                
                if not self.is_stopped():
                    self.progress_update.emit(100, total_size / 1024 / 1024, total_size / 1024 / 1024)
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Download_Complete", file_name=file_name), "green")
                
            except requests.exceptions.RequestException as e:
                write_debug_log(self.get_string("Workers", "DownloaderWorker_Network_Error", e=e))
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Network_Failed"), "red")
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Details", type_e_name=type(e).__name__, e=e), "red")
                all_success = False
                break
            except Exception as e:
                write_debug_log(self.get_string("Workers", "DownloaderWorker_Unexpected_Error", e=e))
                error_msg = self.get_string("Workers", "DownloaderWorker_Error_Unexpected_File_Access")
                self.log_message.emit(error_msg, "red")
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Details", type_e_name=type(e).__name__, e=e), "red")
                all_success = False
                break
        
        self.download_finished.emit(all_success)
        write_debug_log(self.get_string("Workers", "DownloaderWorker_Download_Thread_Exit"))

class TaggerThreadWorker(QObject):
    """Tagging Worker"""
    log_message = Signal(str, str)
    finished = Signal()
    running_state_changed = Signal(bool)
    reload_image_list_signal = Signal()
    def __init__(self, settings: AppSettings, overwrite_checker: Callable[[Path], bool], get_string: Callable[[str, str, Any], str] | None = None, selected_file_path: Path | None = None):
        super().__init__()
        self._settings = settings
        self._overwrite_checker = overwrite_checker
        self._selected_file_path = selected_file_path
        self._mutex = QMutex()
        self._is_stopped = False
        self.get_string = get_string if get_string else lambda section, key, **kwargs: key
    
    def stop(self):
        self._mutex.lock()
        self._is_stopped = True
        self._mutex.unlock()

    def is_stopped(self) -> bool:
        self._mutex.lock()
        stopped = self._is_stopped
        self._mutex.unlock()
        return stopped

    def _mark_model_as_unverified(self):
        """Loads config, sets model as unverified, and saves it."""
        try:
            from config_utils import load_config, load_settings, save_config
            config = load_config()
            settings = load_settings(config)
            if settings.model.verified:
                settings.model.verified = False
                save_config(settings)
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_ModelUnverified"), "orange")
                write_debug_log(self.get_string("Workers", "TaggerThreadWorker_ModelUnverified_Debug"))
        except Exception as e:
            write_debug_log(f"Debug: Failed to save model unverified status: {e}")

    @Slot()
    def run_tagging(self):
        write_debug_log(self.get_string("Workers", "TaggerThreadWorker_Start"))
        self.running_state_changed.emit(True)
        self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Tagging_Process_Start"), "black")
        
        try:
            # Setup Tagger
            tagger, settings_dict = setup_tagger_from_settings(self._settings, self.get_string)
            if not tagger or not settings_dict:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Error_Tagger_Init_Failed"), "red")
                self._mark_model_as_unverified()
                self.running_state_changed.emit(False)
                self.finished.emit()
                return

            # Get image paths
            input_dir = Path(settings_dict['INPUT_DIR'])
            image_paths = get_image_paths_recursive(input_dir)

            # Prioritize the selected file
            if self._selected_file_path and self._selected_file_path in image_paths:
                image_paths.remove(self._selected_file_path)
                image_paths.insert(0, self._selected_file_path)

            if not image_paths:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Warning_No_Image_Files", input_dir=input_dir), "orange")
                self.running_state_changed.emit(False)
                self.finished.emit()
                return

            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Total_Image_Files", count=len(image_paths)), "blue")

            # Define log_to_gui for tagging_core
            def log_to_gui(message: str, color: str):
                write_debug_log(self.get_string("Workers", "TaggerThreadWorker_Core_Log", message=message))
                self.log_message.emit(message, color)

            # Process images
            process_image_loop(
                tagger=tagger,
                image_paths=image_paths,
                settings=settings_dict, # Pass the settings_dict directly
                overwrite_checker=self._overwrite_checker,
                log_gui=log_to_gui,
                stop_checker=self.is_stopped,
                get_string=self.get_string
            )
            
        except Exception as e:
            import traceback
            error_message = self.get_string("Workers", "TaggerThreadWorker_Fatal_Exception", type_e_name=type(e).__name__, e=e, traceback_exc=traceback.format_exc())
            self.log_message.emit(error_message, "red")
            write_debug_log(self.get_string("Workers", "TaggerThreadWorker_Runtime_Exception", e=e, traceback_exc=traceback.format_exc()))
        
        finally:
            self.running_state_changed.emit(False)
            self.reload_image_list_signal.emit()
            self.finished.emit()
            write_debug_log(self.get_string("Workers", "TaggerThreadWorker_Thread_Exit"))

class TagLoader(QObject):
    """Worker to asynchronously load tag files from an image folder"""
    log_message = Signal(str, str)
    tags_loaded = Signal(list)
    finished = Signal()
    def __init__(self, folder: Path, get_string: Callable[[str, str, Any], str] | None = None):
        super().__init__()
        self.folder = folder
        self.get_string = get_string if get_string else lambda section, key, **kwargs: key
        self._mutex = QMutex()
        self._is_stopped = False

    def stop(self):
        self._mutex.lock()
        self._is_stopped = True
        self._mutex.unlock()

    def is_stopped(self):
        self._mutex.lock()
        stopped = self._is_stopped
        self._mutex.unlock()
        return stopped

    def run(self):
        write_debug_log(self.get_string("Workers", "TagLoader_Start", folder=self.folder), get_string=self.get_string)
        counter = Counter()
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
                    write_debug_log(self.get_string("Workers", "TagLoader_TXT_Load_Failed", txt_name=txt.name, e=e), get_string=self.get_string)
            
            if not self.is_stopped():
                all_tags = counter.most_common() 
                self.tags_loaded.emit(all_tags)
        except Exception as e:
            write_debug_log(self.get_string("Workers", "TagLoader_Fatal_Error", e=e), get_string=self.get_string)
            if not self.is_stopped():
                self.tags_loaded.emit([])
        finally:
            self.finished.emit()
            write_debug_log(self.get_string("Workers", "TagLoader_Thread_Exit"), get_string=self.get_string)

class BulkTagWorker(QObject):
    """Worker to execute bulk tag editing (add/delete)"""
    log_message = Signal(str, str)
    finished = Signal()

    def __init__(self, get_string: Callable[[str, str, Any], str] | None = None):
        super().__init__()
        self.get_string = get_string if get_string else lambda section, key, **kwargs: key
        self._mutex = QMutex()
        self._is_stopped = False

    def stop(self):
        self._mutex.lock()
        self._is_stopped = True
        self._mutex.unlock()

    def is_stopped(self):
        self._mutex.lock()
        stopped = self._is_stopped
        self._mutex.unlock()
        return stopped

    @Slot(Path, str)
    def run_bulk_delete(self, input_dir: Path, tag_to_delete: str):
        write_debug_log(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Start", tag_to_delete=tag_to_delete))
        count = 0
        try:
            for txt in input_dir.rglob("*.txt"):
                if self.is_stopped():
                    break
                try:
                    with open(txt, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    tags = [t.strip() for t in content.split(",") if t.strip()]
                    new_tags = [t for t in tags if t != tag_to_delete]
                    
                    if set(tags) != set(new_tags):
                        new_content = ", ".join(new_tags)
                        with open(txt, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        count += 1
                except Exception as e:
                    write_debug_log(self.get_string("Workers", "BulkTagWorker_Tag_Delete_Failed", txt_name=txt.name, e=e))
                    self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Tag_Delete_Failed", txt_name=txt.name), "red")
            
            if not self.is_stopped():
                self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Complete", count=count, tag_to_delete=tag_to_delete), "green")
            write_debug_log(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Count", count=count))
        except Exception as e:
            write_debug_log(self.get_string("Workers", "BulkTagWorker_Unexpected_Error_Bulk_Delete", e=e))
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Unexpected_Bulk_Delete", e=e), "red")
        finally:
            self.finished.emit()
            write_debug_log(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Thread_Exit"))

    @Slot(Path, str, bool)
    def run_bulk_add(self, input_dir: Path, tags_to_add: str, prepend: bool):
        write_debug_log(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Start", tags_to_add=tags_to_add))
        count = 0
        
        # Parse tags to add and create a list without duplicates
        new_tags_to_add = sorted(list(set([t.strip() for t in tags_to_add.split(',') if t.strip()])))
        if not new_tags_to_add:
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Warning_No_Valid_Tags_To_Add"), "orange")
            self.finished.emit()
            return

        try:
            for txt in input_dir.rglob("*.txt"):
                if self.is_stopped():
                    break
                try:
                    with open(txt, "r", encoding="utf-8") as f:
                        existing_tags = [t.strip() for t in f.read().split(',') if t.strip()]

                    if prepend:
                        # Prepend new tags, avoiding duplicates
                        combined_tags = [tag for tag in new_tags_to_add if tag not in existing_tags] + existing_tags
                    else:
                        # Append new tags, avoiding duplicates
                        combined_tags = existing_tags + [tag for tag in new_tags_to_add if tag not in existing_tags]

                    new_content = ", ".join(combined_tags)
                    with open(txt, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    count += 1

                except Exception as e:
                    write_debug_log(self.get_string("Workers", "BulkTagWorker_Tag_Add_Failed", txt_name=txt.name, e=e))
                    self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Tag_Add_Failed", txt_name=txt.name), "red")
            
            if not self.is_stopped():
                self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Complete", count=count, tags_to_add=tags_to_add), "green")
            write_debug_log(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Count", count=count))
        except Exception as e:
            write_debug_log(self.get_string("Workers", "BulkTagWorker_Unexpected_Error_Bulk_Add", e=e))
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Unexpected_Bulk_Add", e=e), "red")
        finally:
            self.finished.emit()
            write_debug_log(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Thread_Exit"))
