from pathlib import Path

def get_txt_path(image_path: Path) -> Path:
    """Return the path to the txt file corresponding to the image path."""
    return image_path.with_suffix('.txt')

def read_tags(txt_path: Path) -> list[str]:
    """Read tags from a txt file and return them as a list."""
    if not txt_path.is_file():
        return []
    try:
        content = txt_path.read_text(encoding='utf-8').strip()
        if not content:
            return []
        # Remove leading/trailing whitespace from tags and exclude empty tags
        return [tag.strip() for tag in content.split(',') if tag.strip()]
    except Exception as e:
        print(f"Error reading tag file {txt_path}: {e}")
        return []

def write_tags(txt_path: Path, tags: list[str]):
    """Write a list of tags to a txt file (overwriting existing content)."""
    # Join tags with ", " as separator
    content = ', '.join(tags)
    try:
        txt_path.write_text(content, encoding='utf-8')
    except Exception as e:
        print(f"Error writing tag file {txt_path}: {e}")

def add_tags_to_file(txt_path: Path, tags_to_add: list[str]) -> bool:
    """Add new tags to an existing tag file (avoiding duplicates)."""
    existing_tags = read_tags(txt_path)
    added = False
    for tag in tags_to_add:
        if tag not in existing_tags:
            existing_tags.append(tag)
            added = True
    
    if added:
        write_tags(txt_path, existing_tags)
    return added

def remove_tag_from_file(txt_path: Path, tag_to_remove: str) -> bool:
    """Remove the specified tag from the tag file."""
    existing_tags = read_tags(txt_path)
    if tag_to_remove in existing_tags:
        existing_tags.remove(tag_to_remove)
        write_tags(txt_path, existing_tags)
        return True
    return False