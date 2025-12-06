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

def load_tag_translation_map(
    english_csv_path: Path, 
    japanese_csv_path: Path, 
    french_csv_path: Path, 
    german_csv_path: Path, 
    spanish_csv_path: Path, 
    russian_csv_path: Path, 
    zh_cn_csv_path: Path, 
    zh_tw_csv_path: Path, 
    korean_csv_path: Path
    ) -> dict[str, list[str]]:
    """
    すべての言語のCSVを読み込み、英語タグをキー、各言語の翻訳リストを値とする辞書を作成します。
    戻り値の形式: { 'english_tag': ['Japanese', 'French', 'German', 'Spanish', 'Russian', 'Zh_CN', 'Zh_TW', 'Korean'] }
    
    CSVファイルの構造:
    - English: id,tag_id,name,category,count,ips (nameが3列目、インデックス2)
    - Japanese: japanese tag (1列目、インデックス0)
    - その他の言語: id,name (nameが2列目、インデックス1)
    """
    mapping: dict[str, list[str]] = {}
    
    # 全てのパスが存在するか確認
    paths = [english_csv_path, japanese_csv_path, french_csv_path, german_csv_path, 
             spanish_csv_path, russian_csv_path, zh_cn_csv_path, zh_tw_csv_path, korean_csv_path]
    
    for p in paths:
        if not p.is_file():
            print(f"File not found: {p}")
            return mapping

    try:
        # 英語版CSVを読み込む（ベースとなる）
        english_tags = []
        with open(english_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダーをスキップ
            for row in reader:
                if len(row) >= 3:
                    # 3列目（インデックス2）がタグ名、アンダースコアをスペースに変換
                    tag = row[2].strip().replace('_', ' ')
                    english_tags.append(tag)
        
        # 各翻訳ファイルを読み込む
        # すべての翻訳ファイルは1列目（インデックス0）に翻訳が格納されている
        translation_files = [
            (japanese_csv_path, 0),
            (french_csv_path, 0),
            (german_csv_path, 0),
            (spanish_csv_path, 0),
            (russian_csv_path, 0),
            (zh_cn_csv_path, 0),
            (zh_tw_csv_path, 0),
            (korean_csv_path, 0)
        ]
        
        all_translations = []
        for trans_path, col_index in translation_files:
            translations = []
            with open(trans_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # ヘッダーをスキップ
                for row in reader:
                    if len(row) > col_index:
                        # カンマで終わる場合は削除
                        trans = row[col_index].strip().rstrip(',')
                        translations.append(trans if trans else '')
                    else:
                        translations.append('')
            all_translations.append(translations)
        
        # 英語タグと翻訳を対応付ける
        for i, en_tag in enumerate(english_tags):
            if en_tag:
                trans_list = []
                for trans_data in all_translations:
                    if i < len(trans_data) and trans_data[i]:
                        trans_list.append(trans_data[i])
                    else:
                        # データがない場合は英語タグをそのまま使用
                        trans_list.append(en_tag)
                mapping[en_tag] = trans_list
                    
    except Exception as e:
        print(f"Error loading tag translations: {e}")
        import traceback
        traceback.print_exc()
        
    return mapping