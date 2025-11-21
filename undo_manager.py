"""
Undo/Redo Manager for Tag Operations
Manages the history of tag editing operations and provides undo/redo functionality.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from utils import write_debug_log


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
