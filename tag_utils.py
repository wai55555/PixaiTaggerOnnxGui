from pathlib import Path
import csv

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

def load_tag_translation_map(english_csv_path: Path, japanese_csv_path: Path) -> dict[str, str]:
    """
    Loads English tags and their Japanese translations, returning a dictionary mapping English -> Japanese.
    Assumes both files have matching lines and the first line is a header to be skipped.
    """
    mapping: dict[str, str] = {}
    if not english_csv_path.is_file() or not japanese_csv_path.is_file():
        return mapping

    try:
        with open(english_csv_path, 'r', encoding='utf-8') as f_en, \
             open(japanese_csv_path, 'r', encoding='utf-8') as f_jp:
            
            reader_en = csv.reader(f_en)
            reader_jp = csv.reader(f_jp)
            
            # Skip header
            try:
                next(reader_en)
                next(reader_jp)
            except StopIteration:
                return mapping

            for row_en, row_jp in zip(reader_en, reader_jp):
                if len(row_en) < 3 or not row_jp:
                    continue
                
                # English tag is at index 2 in selected_tags.csv
                # Replace underscores with spaces to match the format used in the application
                en_tag = row_en[2].strip().replace('_', ' ')
                # Japanese tag is the first column
                jp_tag = row_jp[0].strip()
                
                if en_tag and jp_tag:
                    mapping[en_tag] = jp_tag
                    
    except Exception as e:
        print(f"Error loading tag translations: {e}")
        
    return mapping