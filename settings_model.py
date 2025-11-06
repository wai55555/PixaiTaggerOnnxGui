from dataclasses import dataclass

@dataclass
class Paths:
    input_dir: str
    model_dir: str
    model_filename: str

@dataclass
class Thresholds:
    general: float
    character: float

@dataclass
class Limits:
    general: int
    character: int

@dataclass
class Behavior:
    enable_solo_character_limit: bool
    convert_underscore_to_space: bool


@dataclass
class Window:
    geometry: str
    tag_display_rows: int = 6
    tag_display_cols: int = 5

@dataclass
class Model:
    verified: bool

@dataclass
class Debug:
    debug_log: bool

@dataclass
class AppSettings:
    paths: Paths
    thresholds: Thresholds
    limits: Limits
    behavior: Behavior
    window: Window
    model: Model
    debug: Debug
    language_code: str
