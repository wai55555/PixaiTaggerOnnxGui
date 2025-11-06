from __future__ import annotations

import sys
import csv
import json
import configparser
import traceback
import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping, Sequence, Any, Callable, TYPE_CHECKING
from time import perf_counter
from datetime import datetime

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image
    import onnxruntime as ort
else:
    try:
        import numpy as np
        from PIL import Image
        import onnxruntime as ort
    except ImportError:
        np = None
        Image = None
        ort = None

BASE_DIR = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
LOG_FILE_PATH = BASE_DIR / "debug_log.txt"
CONFIG_PATH = BASE_DIR / "config.ini"

from utils import write_debug_log, log_dbg, nowtag
from settings_model import AppSettings, Paths, Thresholds, Limits, Behavior, Window
from config_utils import load_settings

_get_string = lambda section, key, **kwargs: key # Temporary definition for early checks

if not TYPE_CHECKING:
    if np is None or Image is None or ort is None:
        log_dbg(_get_string("TaggerCore", "Info_Required_Libraries_Missing"))
        if __name__ != "__main__":
            raise ImportError(_get_string("TaggerCore", "Error_Required_Libraries_NotFound"))
        sys.exit(1)

class TagCategory(IntEnum):
    GENERAL = 0
    ARTIST = 1
    RATING = 2
    COPYRIGHT = 3
    CHARACTER = 4
    META = 5

@dataclass(frozen=True)
class TagPrediction:
    name: str
    score: float
    category: TagCategory

@dataclass
class TagResult:
    tags: list[TagPrediction] = field(default_factory=list)
    series_tags: tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class TagMeta:
    name: str
    category: int
    count: int | None = None
    ips: tuple[str, ...] = ()

_CATEGORY_LOOKUP: dict[str, int] = {
    "0": 0, "general": 0,
    "1": 1, "artist": 1,
    "2": 2, "rating": 2,
    "3": 3, "copyright": 3,
    "4": 4, "character": 4,
    "5": 5, "meta": 5,
}

def load_selected_tags(tags_csv: str | Path) -> list[TagMeta]:
    tags_path = Path(tags_csv)
    if not tags_path.is_file():
        log_dbg(_get_string("TaggerCore", "Tag_CSV_File_Not_Found", tags_csv=tags_csv))
        raise FileNotFoundError(_get_string("TaggerCore", "Tag_CSV_File_Not_Found", tags_csv=tags_csv))
    labels: list[TagMeta] = []
    with tags_path.open(encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        try:
            next(reader)
        except StopIteration:
            return []
        for cells in reader:
            if len(cells) < 6:
                continue
            tag_name = cells[2]
            category_str = cells[3]
            count_str = cells[4]
            ips_json = cells[5]
            try:
                category = _CATEGORY_LOOKUP[category_str.lower()]
            except KeyError:
                continue
            try:
                count = int(count_str) if count_str else None
            except ValueError:
                count = None
            ips: tuple[str, ...] = ()
            if ips_json:
                try:
                    parsed = json.loads(ips_json)
                    if isinstance(parsed, list):
                        ips = tuple(parsed)
                except json.JSONDecodeError:
                    pass
            labels.append(TagMeta(name=tag_name, category=category, count=count, ips=ips))
    return labels

def discover_labels_csv(model_dir: Path, tags_csv: str | Path | None) -> Path | None:
    if tags_csv:
        candidate = Path(tags_csv)
        return candidate if candidate.exists() else None
    search_dir = model_dir
    candidates: list[Path] = []
    default_names = ("selected_tags.csv", "selected_tags_v3.csv", "selected_tags_v3c.csv")
    for name in default_names:
        candidate = search_dir / name
        if candidate.is_file() and candidate not in candidates:
             candidates.append(candidate)
    for extra in sorted(search_dir.glob("selected_tags*.csv")):
        if extra not in candidates:
            candidates.append(extra)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def _normalize_np_chw(x: np.ndarray, mean, std) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    for c in range(3):
        x[c] = (x[c] - mean[c]) / std[c]
    return x

class OnnxTagger:
    INPUT_SIZE = 448
    MODEL_MEAN = np.array([0.485, 0.456, 0.406])
    MODEL_STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, model_path: Path, tags_csv: Path | None = None, get_string: Callable[[str, str, Any], str] | None = None):
        self.get_string = get_string if get_string else lambda section, key, **kwargs: key # Store get_string

        if not model_path.is_file():
            log_dbg(self.get_string("TaggerCore", "Model_File_Not_Found", model_path=model_path))
            raise FileNotFoundError(self.get_string("TaggerCore", "Model_File_Not_Found", model_path=model_path))
        if ort is None:
            log_dbg(self.get_string("TaggerCore", "Onnxruntime_Not_Installed"))
            raise ImportError(self.get_string("TaggerCore", "Onnxruntime_Not_Installed"))
        log_dbg(self.get_string("TaggerCore", "Info_ONNX_Session_Creation_Start", model_path=model_path.name))
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        log_dbg(self.get_string("TaggerCore", "Info_ONNX_Session_Created"))
        model_dir = model_path.parent
        tags_path = discover_labels_csv(model_dir, tags_csv)
        if not tags_path or not tags_path.is_file():
            log_dbg(self.get_string("TaggerCore", "Tag_CSV_File_Not_Found_Check_Dir", model_dir=model_dir))
            raise FileNotFoundError(self.get_string("TaggerCore", "Tag_CSV_File_Not_Found_Check_Dir", model_dir=model_dir))
        self.tags = load_selected_tags(tags_path)
        self.tag_meta_lookup: dict[str, TagMeta] = {tag.name: tag for tag in self.tags}
        log_dbg(self.get_string("TaggerCore", "Loaded_Tags_Count", count=len(self.tags), tags_path=tags_path.name))
        self.input_name = self.session.get_inputs()[0].name
        output_names = [output.name for output in self.session.get_outputs()]
        preferred_order = ("prediction", "logits")
        for name in preferred_order:
            if name in output_names:
                self.output_name = name
                break
        else:
            log_dbg(self.get_string("TaggerCore", "ONNX_Prediction_Tensor_NotFound", output_names=output_names))
            raise RuntimeError(self.get_string("TaggerCore", "ONNX_Prediction_Tensor_NotFound", output_names=output_names))

    def prepare_batch_from_rgb_np(self, images: Sequence[np.ndarray]) -> np.ndarray:
        preprocessed_images: list[np.ndarray] = []
        target_size = self.INPUT_SIZE
        for img_array in images:
            image_pil = Image.fromarray(img_array)
            w, h = image_pil.size
            ratio = min(target_size / w, target_size / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            resized_image = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            canvas.paste(resized_image, (x_offset, y_offset))
            img_np = np.asarray(canvas, dtype=np.float32) / 255.0
            img_chw = img_np.transpose((2, 0, 1))
            normalized_chw = _normalize_np_chw(img_chw, self.MODEL_MEAN, self.MODEL_STD)
            preprocessed_images.append(normalized_chw)
        if not preprocessed_images:
            return np.empty((0, 3, target_size, target_size), dtype=np.float32)
        return np.stack(preprocessed_images, axis=0)

    def infer_batch_prepared(self, batch: np.ndarray, *, thresholds: Mapping[Any, float] | None = None, max_tags: Mapping[Any, int] | None = None) -> list[TagResult]:
        if batch.size == 0:
            return []
        input_feed = {self.input_name: batch}
        output = self.session.run([self.output_name], input_feed)[0]
        scores_batch = _sigmoid(output)
        results: list[TagResult] = []
        cat_thresholds = thresholds if thresholds else {}
        cat_limits = max_tags if max_tags else {}
        if not self.tags:
             return [TagResult() for _ in scores_batch]
        hard_cap = sum(cat_limits.values()) if cat_limits else 100
        score_floor = 1e-4
        for scores in scores_batch:
            raw_predictions: list[TagPrediction] = []
            for tag_meta, score in zip(self.tags, scores):
                if score < score_floor:
                    continue
                category = TagCategory(tag_meta.category)
                threshold = cat_thresholds.get(category, cat_thresholds.get(TagCategory.GENERAL, 0.0))
                if float(score) < threshold:
                    continue
                raw_predictions.append(TagPrediction(
                    name=tag_meta.name,
                    score=float(score),
                    category=category
                ))
            ordered = sorted(raw_predictions, key=lambda pred: (-pred.score, pred.name))
            taken: list[TagPrediction] = []
            per_category: dict[TagCategory, int] = {}
            for prediction in ordered:
                if len(taken) >= hard_cap:
                    break
                
                category = prediction.category
                limit = cat_limits.get(category)

                current = per_category.get(category, 0)
                if limit is not None and current >= limit:
                    continue
                
                per_category[category] = current + 1
                taken.append(prediction)
            results.append(TagResult(tags=taken))
        return results

    def infer_batch(self, images: Sequence[Image.Image], *, thresholds: Mapping[Any, float] | None = None, max_tags: Mapping[Any, int] | None = None) -> list[TagResult]:
        rgb_arrays = [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images]
        batch = self.prepare_batch_from_rgb_np(rgb_arrays)
        return self.infer_batch_prepared(batch, thresholds=thresholds, max_tags=max_tags)

def get_image_paths_recursive(base_dir: Path) -> list[Path]:
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(base_dir.rglob(f"*{ext}"))
    return sorted(image_paths)

def format_tags(tag_results: TagResult, convert_underscore: bool) -> str:
    output_tags: list[str] = []
    general_tags: list[TagPrediction] = []
    character_tags: list[TagPrediction] = []
    for pred in tag_results.tags:
        if pred.category == TagCategory.GENERAL:
            general_tags.append(pred)
        elif pred.category == TagCategory.CHARACTER:
            character_tags.append(pred)
    character_tags.sort(key=lambda pred: pred.score, reverse=True)
    for char_pred in character_tags:
        tag_name = char_pred.name
        if convert_underscore:
            tag_name = tag_name.replace("_", " ")
        output_tags.append(tag_name)
    for series_tag in tag_results.series_tags:
        tag_name = series_tag
        if convert_underscore:
            tag_name = tag_name.replace("_", " ")
        output_tags.append(tag_name)
    general_tags.sort(key=lambda pred: pred.score, reverse=True)
    for gen_pred in general_tags:
        tag_name = gen_pred.name
        if convert_underscore:
            tag_name = tag_name.replace("_", " ")
        output_tags.append(tag_name)
    return ", ".join(output_tags)

def setup_tagger_from_settings(app_settings: AppSettings, get_string: Callable[[str, str, Any], str] | None) -> tuple[OnnxTagger | None, dict[str, Any]]:
    """Initializes the tagger and extracts settings from the AppSettings object."""
    try:
        settings_dict = {
            'INPUT_DIR': Path(app_settings.paths.input_dir),
            'MODEL_PATH': BASE_DIR / app_settings.paths.model_dir / app_settings.paths.model_filename,
            'TAG_THRESHOLDS': {
                TagCategory.GENERAL: app_settings.thresholds.general,
                TagCategory.CHARACTER: app_settings.thresholds.character,
            },
            'MAX_TAGS_PER_CATEGORY': {
                TagCategory.GENERAL: app_settings.limits.general,
                TagCategory.CHARACTER: app_settings.limits.character,
            },
            'ENABLE_SOLO_LIMIT': app_settings.behavior.enable_solo_character_limit,
            'CONVERT_UNDERSCORE': app_settings.behavior.convert_underscore_to_space,
        }
        tagger = OnnxTagger(model_path=settings_dict['MODEL_PATH'], get_string=get_string)
        return tagger, settings_dict
    except Exception as e:
        log_dbg(f"Error during Tagger initialization: {type(e).__name__}: {e}")
        log_dbg(f"Traceback: {traceback.format_exc()}")
        return None, {}


def process_image_loop(
    tagger: OnnxTagger,
    settings: dict[str, Any],
    image_paths: list[Path],
    overwrite_checker: Callable[[Path], bool] | None,
    log_gui: Callable[[str, str], None] | None,
    stop_checker: Callable[[], bool] | None,
    get_string: Callable[[str, str, Any], str] | None
):
    """Processes a list of images, applying tags based on the provided settings."""
    core_log_gui = lambda message, color="black": log_gui(message, color) if log_gui else None
    _get_string = get_string if get_string else lambda section, key, **kwargs: key

    for i, image_path in enumerate(image_paths):
        if stop_checker and stop_checker():
            core_log_gui(_get_string("TaggerCore", "Tagging_Process_Aborted_By_User"), "red")
            log_dbg(_get_string("TaggerCore", "Tagging_Process_Aborted_By_User_Debug"))
            break

        relative_path = image_path.relative_to(settings['INPUT_DIR'])
        current_index_str = f"[{i+1}/{len(image_paths)}]"
        core_log_gui(_get_string("TaggerCore", "Processing_Image", current_index_str=current_index_str, relative_path=relative_path), "black")
        
        try:
            with open(image_path, 'rb') as f:
                image = Image.open(f).convert("RGB")
        except Exception as e:
            log_dbg(_get_string("TaggerCore", "Image_Load_Failed", current_index_str=current_index_str, relative_path=relative_path, type_e_name=type(e).__name__, e=e))
            core_log_gui(_get_string("TaggerCore", "Image_Load_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "red")
            continue

        try:
            results = tagger.infer_batch(
                images=[image],
                thresholds=settings['TAG_THRESHOLDS'],
                max_tags=settings['MAX_TAGS_PER_CATEGORY'],
            )
        except Exception as e:
            log_dbg(_get_string("TaggerCore", "Tag_Inference_Failed", current_index_str=current_index_str, relative_path=relative_path, type_e_name=type(e).__name__, e=e))
            core_log_gui(_get_string("TaggerCore", "Tag_Inference_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "red")
            continue

        if not results or not results[0].tags:
            log_dbg(_get_string("TaggerCore", "Tag_Acquisition_Failed", current_index_str=current_index_str, relative_path=relative_path))
            core_log_gui(_get_string("TaggerCore", "Tag_Acquisition_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "orange")
            continue

        tag_result = results[0]
        
        final_tags, all_series_tags = filter_tags_by_solo_rule(
            tag_result, tagger, settings['ENABLE_SOLO_LIMIT']
        )
        tag_result.tags = final_tags
        tag_result.series_tags = tuple(sorted(list(all_series_tags)))
        
        formatted_tags = format_tags(tag_result, settings['CONVERT_UNDERSCORE'])
        base_name, _ = os.path.splitext(str(image_path))
        output_path = Path(base_name + ".txt")

        if output_path.is_file() and overwrite_checker and not overwrite_checker(output_path):
            log_dbg(_get_string("TaggerCore", "Tag_Output_Skipped_Existing_File", current_index_str=current_index_str, relative_path=relative_path))
            
            core_log_gui(_get_string("TaggerCore", "Tag_Skipped_Existing_File_Short", current_index_str=current_index_str, output_path_name=output_path.name), "orange")
            continue

        try:
            long_path_str = str(output_path.resolve())
            if sys.platform == "win32":
                long_path_str = "\\\\?\\" + long_path_str
            with open(long_path_str, 'w', encoding='utf-8') as f:
                f.write(formatted_tags)

            core_log_gui(_get_string("TaggerCore", "Tag_Output_Success", current_index_str=current_index_str, output_path_name=output_path.name), "green")
            log_dbg(_get_string("TaggerCore", "Tagging_Result_Output", current_index_str=current_index_str, output_path_name=output_path.name))
        except Exception as e:
            log_dbg(_get_string("TaggerCore", "Save_Failed", current_index_str=current_index_str, relative_path=relative_path, type_e_name=type(e).__name__, e=e))
            
            core_log_gui(_get_string("TaggerCore", "Save_Failed_Short", current_index_str=current_index_str, output_path_name=output_path.name), "red")

def filter_tags_by_solo_rule(
    tag_result: TagResult,
    tagger: OnnxTagger,
    enable_solo_limit: bool
) -> tuple[list[TagPrediction], set[str]]:
    """
    if solo tag is available.    
    Returns:
        (final_tags, all_series_tags)
    """
    general_tags = [pred for pred in tag_result.tags if pred.category == TagCategory.GENERAL]
    character_tags = [pred for pred in tag_result.tags if pred.category == TagCategory.CHARACTER]
    solo_tag_found = any(pred.name.lower() == "solo" for pred in general_tags)
    all_series_tags: set[str] = set()
    
    # if solo tag is available, and a character tag exists
    if enable_solo_limit and solo_tag_found and character_tags:
        # Only keep the character tag with the highest score.
        character_tags.sort(key=lambda pred: pred.score, reverse=True)
        best_character_tag = character_tags[0]
        
        # get series tags from the best character tag
        char_meta = tagger.tag_meta_lookup.get(best_character_tag.name)
        if char_meta and char_meta.ips:
            all_series_tags.update(char_meta.ips)
        
        final_tags = general_tags + [best_character_tag]
    else:
        # if solo tag not found
        for char_pred in character_tags:
            char_meta = tagger.tag_meta_lookup.get(char_pred.name)
            if char_meta and char_meta.ips:
                all_series_tags.update(char_meta.ips)
        
        final_tags = general_tags + character_tags
    
    return final_tags, all_series_tags

def main(overwrite_checker: Callable[[Path], bool] | None = None, log_gui: Callable[[str, str], None] | None = None, stop_checker: Callable[[], bool] | None = None, get_string: Callable[[str, str, Any], str] | None = None):
    log_dbg(_get_string("TaggerCore", "Info_Tagging_Core_Main_Start"))
    start_time = perf_counter()
    core_log_gui = lambda message, color="black": log_gui(message, color) if log_gui else None
    _get_string = get_string if get_string else lambda section, key, **kwargs: key

    try:
        assert CONFIG_PATH.is_file(), _get_string("TaggerCore", "Config_File_NotFound", CONFIG_PATH=CONFIG_PATH)
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH, encoding='utf-8')
    except Exception as e:
        log_dbg(_get_string("TaggerCore", "Fatal_Error_Config_Load_Failed", type_e_name=type(e).__name__, e=e))
        core_log_gui(_get_string("TaggerCore", "Fatal_Error_Config_Load_Failed_GUI"), "red")
        return

    # Load AppSettings from config
    app_settings = load_settings(config)

    tagger, settings_dict = setup_tagger_from_settings(app_settings, _get_string)
    if not tagger or not settings_dict:
        core_log_gui(_get_string("TaggerCore", "Error_Tagger_Init_Failed_GUI"), "red")
        return

    image_paths = get_image_paths_recursive(settings_dict['INPUT_DIR'])
    if not image_paths:
        log_dbg(_get_string("TaggerCore", "Warning_No_Image_Files_Found_In_Dir", INPUT_DIR=settings_dict['INPUT_DIR']))
        core_log_gui(_get_string("TaggerCore", "Warning_No_Image_Files_Found_GUI"), "orange")
        return

    core_log_gui(_get_string("TaggerCore", "Total_Image_Files_Found", count=len(image_paths)), "blue")
    
    process_image_loop(tagger, settings_dict, image_paths, overwrite_checker, log_gui, stop_checker, _get_string)

    end_time = perf_counter()
    log_dbg(_get_string("TaggerCore", "Total_Processing_Time", time=f"{end_time - start_time:.2f}"))
    core_log_gui(_get_string("TaggerCore", "Tagging_Process_Complete"), "green")

if __name__ == "__main__":
    main()