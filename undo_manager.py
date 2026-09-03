"""
Undo/Redo Manager for Tag Operations
Manages the history of tag editing operations and provides undo/redo functionality.
"""

from __future__ import annotations
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from utils import write_debug_log


def _long_path_str(path: Path) -> str:
    """Windows の MAX_PATH（260文字）制限を避ける `\\\\?\\` プレフィックス付き絶対パス。

    tagging_core.process_image_loop / caption_core.process_caption_loop の書き込みと
    同じ狙い（長いパスでも書けるのに undo/redo だけ素の Path 経由で失敗する非対称を防ぐ）。

    - `os.path.abspath()` を使う（`Path.resolve()` ではない）: `\\\\?\\` パスは OS がそのまま
      使うので `..` の正規化は必要だが、**symlink は解決しない**。resolve() だと、新規
      作成した出力がその後 symlink に差し替えられていた場合、undo の unlink が
      リンク先の実体を消してしまう（PR#16 レビュー指摘）。
    - UNC パス（`\\\\server\\share\\...`）は `\\\\?\\UNC\\server\\share\\...` の形にする必要が
      あり、単純に `\\\\?\\` を前置すると不正なパスになる（PR#16 レビュー指摘）。
    - 既に `\\\\?\\` 付き（拡張長パス）が渡された場合はそのまま返す。二重に前置すると
      `\\\\?\\UNC\\?\\...` のような不正パスになる（PR#16 レビュー第2ラウンド指摘）。
    """
    p = os.path.abspath(path)
    if sys.platform != "win32":
        return p
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC" + p[1:]
    return "\\\\?\\" + p


class GetString(Protocol):
    """Protocol for localization function."""
    def __call__(self, section: str, key: str, **kwargs: str | int | float) -> str:
        ...


class UndoAction(ABC):
    """Abstract base class for all undoable actions."""
    
    @abstractmethod
    def undo(self) -> bool:
        """
        Undo this action.
        Returns True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def redo(self) -> bool:
        """
        Redo this action.
        Returns True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def description(self) -> str:
        """
        Returns a human-readable description of this action.
        """
        pass


@dataclass
class AddTagsAction(UndoAction):
    """Action for adding tags to a single image."""
    file_path: Path
    added_tags: list[str]
    
    def undo(self) -> bool:
        """Remove the added tags from the file."""
        try:
            if not self.file_path.exists():
                write_debug_log(f"Undo failed: File not found: {self.file_path}")
                return False
            
            # Read current tags
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                return True
            
            tags = [tag.strip() for tag in content.split(',')]
            
            # Remove added tags
            for tag in self.added_tags:
                if tag in tags:
                    tags.remove(tag)
            
            # Write back
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(', '.join(tags))
            
            write_debug_log(f"Undo AddTags: Removed {len(self.added_tags)} tags from {self.file_path.name}")
            return True
            
        except Exception as e:
            write_debug_log(f"Undo AddTags failed: {e}")
            return False
    
    def redo(self) -> bool:
        """Re-add the tags to the file."""
        try:
            if not self.file_path.exists():
                write_debug_log(f"Redo failed: File not found: {self.file_path}")
                return False
            
            # Read current tags
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            tags = [tag.strip() for tag in content.split(',')] if content else []
            
            # Add tags (avoid duplicates)
            for tag in self.added_tags:
                if tag not in tags:
                    tags.append(tag)
            
            # Write back
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(', '.join(tags))
            
            write_debug_log(f"Redo AddTags: Added {len(self.added_tags)} tags to {self.file_path.name}")
            return True
            
        except Exception as e:
            write_debug_log(f"Redo AddTags failed: {e}")
            return False
    
    def description(self) -> str:
        """Return a description of this action."""
        if len(self.added_tags) == 1:
            return f"「{self.added_tags[0]}」の追加"
        elif len(self.added_tags) <= 3:
            return f"「{', '.join(self.added_tags)}」の追加"
        else:
            return f"「{', '.join(self.added_tags[:3])}...」など{len(self.added_tags)}個のタグの追加"


@dataclass
class RemoveTagAction(UndoAction):
    """Action for removing a tag from a single image."""
    file_path: Path
    removed_tag: str
    original_index: int
    
    def undo(self) -> bool:
        """Re-insert the removed tag at its original position."""
        try:
            if not self.file_path.exists():
                write_debug_log(f"Undo failed: File not found: {self.file_path}")
                return False
            
            # Read current tags
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            tags = [tag.strip() for tag in content.split(',')] if content else []
            
            # Insert tag at original position
            insert_pos = min(self.original_index, len(tags))
            tags.insert(insert_pos, self.removed_tag)
            
            # Write back
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(', '.join(tags))
            
            write_debug_log(f"Undo RemoveTag: Re-inserted '{self.removed_tag}' at position {insert_pos} in {self.file_path.name}")
            return True
            
        except Exception as e:
            write_debug_log(f"Undo RemoveTag failed: {e}")
            return False
    
    def redo(self) -> bool:
        """Remove the tag again."""
        try:
            if not self.file_path.exists():
                write_debug_log(f"Redo failed: File not found: {self.file_path}")
                return False
            
            # Read current tags
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                return True
            
            tags = [tag.strip() for tag in content.split(',')]
            
            # Remove tag
            if self.removed_tag in tags:
                tags.remove(self.removed_tag)
            
            # Write back
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(', '.join(tags))
            
            write_debug_log(f"Redo RemoveTag: Removed '{self.removed_tag}' from {self.file_path.name}")
            return True
            
        except Exception as e:
            write_debug_log(f"Redo RemoveTag failed: {e}")
            return False
    
    def description(self) -> str:
        """Return a description of this action."""
        return f"「{self.removed_tag}」の削除"


@dataclass
class EditCaptionAction(UndoAction):
    """Action for editing an image's free-text caption (captioner models).

    Unlike the tag actions this stores the whole before/after text - captions are
    short and not comma-structured, so a positional diff would not help.
    """
    file_path: Path
    old_text: str
    new_text: str
    file_existed_before: bool = True

    def _write(self, text: str) -> bool:
        try:
            self.file_path.write_text(text, encoding='utf-8')
            return True
        except Exception as e:
            write_debug_log(f"EditCaption write failed for {self.file_path}: {e}")
            return False

    def undo(self) -> bool:
        if not self.file_existed_before:
            # The edit created the file; undo must restore its absence, otherwise
            # caption generation treats the leftover .txt as done and skips the image.
            try:
                self.file_path.unlink(missing_ok=True)
                return True
            except Exception as e:
                write_debug_log(f"EditCaption undo unlink failed for {self.file_path}: {e}")
                return False
        return self._write(self.old_text)

    def redo(self) -> bool:
        return self._write(self.new_text)

    def description(self) -> str:
        return "キャプションの編集"


@dataclass
class _FileSnapshotAction(UndoAction):
    """全文スナップショットによる undo/redo の共通実装（PR#16 レビュー指摘: 従来
    OverwriteFileAction と AppendTagsActionV2 が完全に同じ実装を重複させていた）。

    previous_content が None のときは「変更前にファイルが存在しなかった」ことを表し、
    undo はファイル削除まで行う。undo / redo とも全文スナップショットの復元で対称。
    `description()` はサブクラスで実装する（抽象のまま = 直接インスタンス化はしない）。
    """
    file_path: Path
    previous_content: str | None
    new_content: str

    def _write(self, text: str) -> bool:
        try:
            with open(_long_path_str(self.file_path), 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        except Exception as e:
            write_debug_log(f"{type(self).__name__} write failed for {self.file_path}: {e}")
            return False

    def undo(self) -> bool:
        if self.previous_content is None:
            try:
                os.unlink(_long_path_str(self.file_path))
            except FileNotFoundError:
                pass
            except Exception as e:
                write_debug_log(f"{type(self).__name__} undo unlink failed for {self.file_path}: {e}")
                return False
            return True
        return self._write(self.previous_content)

    def redo(self) -> bool:
        return self._write(self.new_content)


@dataclass
class OverwriteFileAction(_FileSnapshotAction):
    """タグ付けバッチによる1ファイルの上書き / 新規作成（spec.md 4.1節）。"""

    def description(self) -> str:
        return f"「{self.file_path.name}」の上書き"


@dataclass
class AppendTagsActionV2(_FileSnapshotAction):
    """タグ付けバッチによる1ファイルへの追記（spec.md 4.1節）。

    undo/redo は _FileSnapshotAction 共通の全文スナップショット方式で行い、
    added_tags は説明文・ログ用のメタ情報として保持する。
    """
    added_tags: list[str]

    def description(self) -> str:
        return f"「{self.file_path.name}」へ{len(self.added_tags)}件のタグを追記"


@dataclass
class CompositeUndoAction(UndoAction):
    """バッチ1回分の複数ファイル変更を、Undo履歴上の1エントリとして扱う（spec.md 4.2節）。

    undo は逆順、redo は元の順で実行する。max_history=50 が大量画像処理で
    即座に消費されるのを防ぐのが目的。
    """
    actions: list[UndoAction]
    label: str = ""

    def undo(self) -> bool:
        # 逆順に戻す。1つでも成功していれば「元に戻した」とみなす（部分成功を許容）
        results = [action.undo() for action in reversed(self.actions)]
        return any(results)

    def redo(self) -> bool:
        results = [action.redo() for action in self.actions]
        return any(results)

    def description(self) -> str:
        if self.label:
            return self.label
        return f"タグ付けによる{len(self.actions)}ファイルの変更"


@dataclass
class BulkAddTagsAction(UndoAction):
    """Action for adding tags to multiple images."""
    file_paths: list[Path]
    added_tags: list[str]
    position: str  # "prepend" or "append"
    
    def undo(self) -> bool:
        """Remove the added tags from all files."""
        success_count = 0
        for file_path in self.file_paths:
            try:
                if not file_path.exists():
                    write_debug_log(f"Undo: File not found: {file_path}")
                    continue
                
                # Read current tags
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    success_count += 1
                    continue
                
                tags = [tag.strip() for tag in content.split(',')]
                
                # Remove added tags
                for tag in self.added_tags:
                    if tag in tags:
                        tags.remove(tag)
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(', '.join(tags))
                
                success_count += 1
                
            except Exception as e:
                write_debug_log(f"Undo BulkAddTags failed for {file_path}: {e}")
        
        write_debug_log(f"Undo BulkAddTags: Processed {success_count}/{len(self.file_paths)} files")
        return success_count > 0
    
    def redo(self) -> bool:
        """Re-add the tags to all files."""
        success_count = 0
        for file_path in self.file_paths:
            try:
                if not file_path.exists():
                    write_debug_log(f"Redo: File not found: {file_path}")
                    continue
                
                # Read current tags
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                tags = [tag.strip() for tag in content.split(',')] if content else []
                
                # Add tags based on position
                if self.position == "prepend":
                    # Add to beginning (avoid duplicates)
                    for tag in reversed(self.added_tags):
                        if tag not in tags:
                            tags.insert(0, tag)
                else:  # append
                    # Add to end (avoid duplicates)
                    for tag in self.added_tags:
                        if tag not in tags:
                            tags.append(tag)
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(', '.join(tags))
                
                success_count += 1
                
            except Exception as e:
                write_debug_log(f"Redo BulkAddTags failed for {file_path}: {e}")
        
        write_debug_log(f"Redo BulkAddTags: Processed {success_count}/{len(self.file_paths)} files")
        return success_count > 0
    
    def description(self) -> str:
        """Return a description of this action."""
        if len(self.added_tags) == 1:
            return f"「{self.added_tags[0]}」の一括追加（{len(self.file_paths)}ファイル）"
        else:
            return f"{len(self.added_tags)}個のタグの一括追加（{len(self.file_paths)}ファイル）"


@dataclass
class BulkRemoveTagsAction(UndoAction):
    """Action for removing a tag from multiple images."""
    removed_tag: str
    file_tag_positions: list[tuple[Path, int]]  # (file_path, original_index)
    
    def undo(self) -> bool:
        """Re-insert the removed tag at its original position in each file."""
        success_count = 0
        for file_path, original_index in self.file_tag_positions:
            try:
                if not file_path.exists():
                    write_debug_log(f"Undo: File not found: {file_path}")
                    continue
                
                # Read current tags
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                tags = [tag.strip() for tag in content.split(',')] if content else []
                
                # Insert tag at original position
                insert_pos = min(original_index, len(tags))
                tags.insert(insert_pos, self.removed_tag)
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(', '.join(tags))
                
                success_count += 1
                
            except Exception as e:
                write_debug_log(f"Undo BulkRemoveTags failed for {file_path}: {e}")
        
        write_debug_log(f"Undo BulkRemoveTags: Processed {success_count}/{len(self.file_tag_positions)} files")
        return success_count > 0
    
    def redo(self) -> bool:
        """Remove the tag again from all files."""
        success_count = 0
        for file_path, _ in self.file_tag_positions:
            try:
                if not file_path.exists():
                    write_debug_log(f"Redo: File not found: {file_path}")
                    continue
                
                # Read current tags
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    success_count += 1
                    continue
                
                tags = [tag.strip() for tag in content.split(',')]
                
                # Remove tag
                if self.removed_tag in tags:
                    tags.remove(self.removed_tag)
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(', '.join(tags))
                
                success_count += 1
                
            except Exception as e:
                write_debug_log(f"Redo BulkRemoveTags failed for {file_path}: {e}")
        
        write_debug_log(f"Redo BulkRemoveTags: Processed {success_count}/{len(self.file_tag_positions)} files")
        return success_count > 0
    
    def description(self) -> str:
        """Return a description of this action."""
        return f"「{self.removed_tag}」の一括削除（{len(self.file_tag_positions)}ファイル）"


class UndoManager:
    """Manages the history of undoable actions."""
    
    def __init__(self, max_history: int = 50):
        """
        Initialize the UndoManager.
        
        Args:
            max_history: Maximum number of actions to keep in history.
        """
        self.undo_stack: list[UndoAction] = []
        self.redo_stack: list[UndoAction] = []
        self.max_history = max_history
        write_debug_log(f"UndoManager initialized with max_history={max_history}")
    
    def push(self, action: UndoAction) -> None:
        """
        Add a new action to the undo stack.
        This clears the redo stack.
        
        Args:
            action: The action to add.
        """
        self.undo_stack.append(action)
        self.redo_stack.clear()
        
        # Limit history size
        if len(self.undo_stack) > self.max_history:
            removed = self.undo_stack.pop(0)
            write_debug_log(f"History limit reached, removed oldest action: {removed.description()}")
        
        write_debug_log(f"Action pushed: {action.description()} (undo_stack size: {len(self.undo_stack)})")
    
    def can_undo(self) -> bool:
        """Check if there are actions that can be undone."""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if there are actions that can be redone."""
        return len(self.redo_stack) > 0
    
    def undo(self) -> bool:
        """
        Undo the most recent action.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.can_undo():
            write_debug_log("Undo failed: No actions to undo")
            return False
        
        action = self.undo_stack.pop()
        write_debug_log(f"Undoing: {action.description()}")
        
        if action.undo():
            self.redo_stack.append(action)
            write_debug_log(f"Undo successful (redo_stack size: {len(self.redo_stack)})")
            return True
        else:
            write_debug_log("Undo failed, action not added to redo stack")
            return False
    
    def redo(self) -> bool:
        """
        Redo the most recently undone action.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.can_redo():
            write_debug_log("Redo failed: No actions to redo")
            return False
        
        action = self.redo_stack.pop()
        write_debug_log(f"Redoing: {action.description()}")
        
        if action.redo():
            self.undo_stack.append(action)
            write_debug_log(f"Redo successful (undo_stack size: {len(self.undo_stack)})")
            return True
        else:
            write_debug_log("Redo failed, action not added to undo stack")
            return False
    
    def get_undo_description(self) -> str:
        """
        Get a description of the next action that would be undone.
        
        Returns:
            Description string, or empty string if no actions to undo.
        """
        if self.can_undo():
            return self.undo_stack[-1].description()
        return ""
    
    def get_redo_description(self) -> str:
        """
        Get a description of the next action that would be redone.
        
        Returns:
            Description string, or empty string if no actions to redo.
        """
        if self.can_redo():
            return self.redo_stack[-1].description()
        return ""
    
    def clear(self) -> None:
        """Clear all undo/redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        write_debug_log("Undo/Redo history cleared")
