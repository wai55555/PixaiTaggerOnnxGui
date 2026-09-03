"""TargetMode shared filter unit tests.

Offline only - no network, no GUI. Run:  rtk pytest tests/test_target_select.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tagging_core import TargetMode, filter_target_images, parse_target_mode


def _make_fixture(tmp_path):
    with_txt = tmp_path / "with_txt.jpg"
    without_txt = tmp_path / "without_txt.jpg"
    other = tmp_path / "other.jpg"
    for p in (with_txt, without_txt, other):
        p.write_bytes(b"img")
    (tmp_path / "with_txt.txt").write_text("already tagged", encoding="utf-8")
    paths = [with_txt, without_txt, other]
    return paths, with_txt, without_txt, other


def test_all_passthrough(tmp_path):
    paths, with_txt, without_txt, other = _make_fixture(tmp_path)
    assert filter_target_images(paths, TargetMode.ALL) == paths
    print("  ALL passthrough: OK")


def test_unprocessed_only_missing_txt(tmp_path):
    paths, with_txt, without_txt, other = _make_fixture(tmp_path)
    assert filter_target_images(paths, TargetMode.UNPROCESSED) == [without_txt, other]
    print("  UNPROCESSED missing-txt only: OK")


def test_unprocessed_empty_txt_counts_as_processed(tmp_path):
    paths, with_txt, without_txt, other = _make_fixture(tmp_path)
    empty_txt = tmp_path / "other.txt"
    empty_txt.write_text("", encoding="utf-8")
    assert filter_target_images(paths, TargetMode.UNPROCESSED) == [without_txt]
    print("  UNPROCESSED empty-txt is processed: OK")


def test_failed_intersection_order_preserved(tmp_path):
    paths, with_txt, without_txt, other = _make_fixture(tmp_path)
    failed = [other, with_txt]
    assert filter_target_images(paths, TargetMode.FAILED, failed_paths=failed) == [with_txt, other]
    assert filter_target_images(paths, TargetMode.FAILED, failed_paths=()) == []
    print("  FAILED intersection order preserved: OK")


def test_selected_inside_returns_single(tmp_path):
    paths, with_txt, without_txt, other = _make_fixture(tmp_path)
    assert filter_target_images(paths, TargetMode.SELECTED, selected=without_txt) == [without_txt]
    print("  SELECTED inside: OK")


def test_selected_none_returns_empty(tmp_path):
    paths, with_txt, without_txt, other = _make_fixture(tmp_path)
    assert filter_target_images(paths, TargetMode.SELECTED, selected=None) == []
    print("  SELECTED None: OK")


def test_selected_out_of_input_returns_empty(tmp_path):
    paths, with_txt, without_txt, other = _make_fixture(tmp_path)
    outsider = tmp_path / "outsider.jpg"
    outsider.write_bytes(b"img")
    assert filter_target_images(paths, TargetMode.SELECTED, selected=outsider) == []
    print("  SELECTED out-of-input: OK")


def test_parse_invalid_returns_all():
    assert parse_target_mode("bogus-value") is TargetMode.ALL
    assert parse_target_mode("") is TargetMode.ALL
    assert parse_target_mode("all") is TargetMode.ALL
    assert parse_target_mode("UNPROCESSED") is TargetMode.UNPROCESSED
    assert parse_target_mode("failed") is TargetMode.FAILED
    assert parse_target_mode(" Selected ") is TargetMode.SELECTED
    print("  parse invalid=ALL: OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TARGET SELECT TESTS PASSED")
