"""キャプション保存（260901_VLM_spec.md 11章 / design.md 8章）。

- 既存内容とキャプションの区切りは改行1つ（既存 caption_core の ", " 区切りとは別物）
- APPEND / PREPEND では完全一致の重複追加を防ぐ
- 一時ファイル + 検証 + os.replace の原子的書き込み。失敗時は既存 .txt を変えない
- 生成失敗時は保存処理を呼ばない（呼び出し側の責務）
"""
from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


# 既存 tagging_core / caption_core と同じ拡張長パス変換（undo_manager._long_path_str と同趣旨）。
def _long_path_str(path: Path) -> str:
    p = os.path.abspath(path)
    if sys.platform != "win32":
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC" + p[1:]
    return "\\\\?\\" + p


VLM_PLACEMENTS = ("PREPEND", "APPEND", "OVERWRITE")


def parse_placement(raw: str) -> str:
    v = str(raw).strip().upper()
    return v if v in VLM_PLACEMENTS else "OVERWRITE"


def combine_caption(existing: str, caption: str, placement: str) -> str:
    """既存内容と生成キャプションを改行1つで結合する（spec.md 11.1節）。

    片方が空なら余分な改行を足さない。OVERWRITE は既存を無視してキャプションのみ。
    """
    placement = parse_placement(placement)
    existing = existing.strip()
    caption = caption.strip()
    if placement == "OVERWRITE" or not existing:
        return caption
    if not caption:
        return existing
    if placement == "PREPEND":
        return f"{caption}\n{existing}"
    return f"{existing}\n{caption}"


def caption_already_present(existing: str, caption: str) -> bool:
    """既存内容の中に、追加しようとするキャプションと完全一致する塊があるか。

    意味的・表現的な類似は判定しない（spec.md 11.3節）。複数行キャプションにも対応する
    よう、改行を境界とした contiguous な完全一致（先頭 / 末尾 / 中間 / 全体）を見る。
    """
    target = caption.strip()
    ex = existing.strip()
    if not target:
        return True
    if ex == target:
        return True
    return (ex.startswith(target + "\n")
            or ex.endswith("\n" + target)
            or ("\n" + target + "\n") in ex)


@dataclass(frozen=True)
class SaveOutcome:
    written: bool
    path: Path
    previous_content: str | None      # None = 変更前にファイルが存在しなかった
    new_content: str
    skipped_reason: str = ""          # "duplicate" / "no_change" / ""（＝書き込んだ）


def save_caption(output_path: Path, caption: str, placement: str) -> SaveOutcome:
    """キャプションを output_path へ保存する。

    手順（spec.md 11.2節）:
      1. 既存ファイルを読み込む（読めない既存ファイルは触らず例外を送出）
      2. 結合後の内容をメモリで作る
      3. 一時ファイルへ書く
      4. 読み直して検証する
      5. os.replace で設置する
    """
    placement = parse_placement(placement)
    caption = caption.strip()
    if not caption:
        raise ValueError("empty caption")

    previous_content: str | None = None
    if output_path.is_file():
        # 読めない既存ファイルは新規扱いにしない（undo での破壊防止。PR#16 と同方針）。
        previous_content = output_path.read_text(encoding="utf-8")

    if placement in ("APPEND", "PREPEND") and previous_content is not None:
        if caption_already_present(previous_content, caption):
            return SaveOutcome(False, output_path, previous_content,
                               previous_content, skipped_reason="duplicate")

    new_content = combine_caption(previous_content or "", caption, placement)
    if previous_content is not None and new_content == previous_content:
        return SaveOutcome(False, output_path, previous_content, previous_content,
                           skipped_reason="no_change")

    _atomic_write(output_path, new_content)
    return SaveOutcome(True, output_path, previous_content, new_content)


def _atomic_write(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f"{output_path.stem}.{uuid.uuid4().hex}.vlmtmp")
    try:
        with open(_long_path_str(tmp), "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # 検証: 書いた内容が読み直せて一致するか。newline="" で改行変換を無効化する
        # （universal newlines だと content に \r\n が含まれる場合に誤って不一致になる）。
        with open(_long_path_str(tmp), "r", encoding="utf-8", newline="") as f:
            if f.read() != content:
                raise OSError("verification mismatch after write")
        os.replace(_long_path_str(tmp), _long_path_str(output_path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
