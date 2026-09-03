"""[Behavior] target_mode の config 往復テスト（260903_vlm-gap-fix.md todo 5）.

save 側は dataclass の汎用走査で自動保存されるが、load 側は明示 loader へ
1行足さないと復元されない（Oracle レビュー指摘）。ここで load 明示 loader +
既定値 + save 往復を固定する。

Offline only - no network, no GUI. Run:  rtk pytest tests/test_target_mode_config.py -q
"""
import configparser
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app_settings as A


def test_default_is_all():
    s = A.load_settings(A.get_default_config())
    assert s.behavior.target_mode == "ALL"
    print("  default target_mode == ALL: OK")


def test_old_config_without_key_falls_back_to_all():
    cfg = configparser.ConfigParser()
    cfg.read_dict({"Behavior": {"enable_solo_character_limit": "True",
                                "convert_underscore_to_space": "True",
                                "existing_file_mode": "ASK"}})
    s = A.load_settings(cfg)
    assert s.behavior.target_mode == "ALL"
    print("  missing key -> ALL (old config unchanged): OK")


def test_invalid_value_falls_back_to_all():
    cfg = A.get_default_config()
    cfg.set("Behavior", "target_mode", "bogus")
    s = A.load_settings(cfg)
    assert s.behavior.target_mode == "ALL"
    assert A.parse_target_mode_setting("") == "ALL"
    assert A.parse_target_mode_setting("failed") == "FAILED"   # case-insensitive
    print("  invalid -> ALL, case-insensitive parse: OK")


def test_save_load_round_trip(monkeypatch):
    tmp = Path(tempfile.mkdtemp()) / "config.ini"
    monkeypatch.setattr(A, "CONFIG_PATH", tmp)

    s = A.load_settings(A.get_default_config())
    s.behavior.target_mode = "FAILED"
    A.save_config(s)

    written = configparser.ConfigParser()
    written.read(tmp, encoding="utf-8")
    assert written.get("Behavior", "target_mode") == "FAILED"

    s2 = A.load_settings(written)
    assert s2.behavior.target_mode == "FAILED"
    print("  save -> config.ini -> load preserves FAILED: OK")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
