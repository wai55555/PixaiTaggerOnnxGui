# -*- mode: python ; coding: utf-8 -*-

import ast
import glob
import os
import re

# Ship every model's hand-authored model_config.json (NOT the multi-GB model.onnx files,
# which the app downloads at runtime) plus PixAI's curated tag-translation CSVs, keeping
# the models/<model_id>/ directory structure. constants._seed_bundled_model_files() copies
# them out of _internal/ into the user-visible models/ folder on first launch.
_model_datas = [
    (p, os.path.dirname(p))
    for p in glob.glob('models/*/model_config.json')
]


def _translation_suffixes():
    """The languages tag_utils actually loads, read straight from the source so a new
    language never silently misses the build. Falls back to the current list."""
    fallback = ["jp", "fr", "de", "es", "ru", "zh_CN", "zh_TW", "ko"]
    try:
        src = open('tag_utils.py', encoding='utf-8').read()
        match = re.search(r'_TRANSLATION_LANGUAGE_SUFFIXES\s*=\s*(\[[^\]]*\])', src)
        return ast.literal_eval(match.group(1)) if match else fallback
    except Exception:
        return fallback


# Only the 8 hand-curated translation CSVs. A bare `selected_tags*.csv` glob would also
# sweep up selected_tags.csv (downloaded at runtime) and selected_tags_en.csv (redundant),
# both gitignored - that would make the build depend on the developer's local downloads.
_pixai_dir = 'models/pixai-tagger-v0.9'
_model_datas += [
    (os.path.join(_pixai_dir, f'selected_tags_{suffix}.csv'), _pixai_dir)
    for suffix in _translation_suffixes()
    if os.path.isfile(os.path.join(_pixai_dir, f'selected_tags_{suffix}.csv'))
]


a = Analysis(
    ['pixai_tagger_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('icons', 'icons'), ('lang', 'lang')] + _model_datas,
    hiddenimports=[
        'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
        # keyring はバックエンドを entry point 経由で探すため、凍結ビルドでは
        # 明示しないと 1つも見つからず、API キーが毎回セッション保持に落ちる。
        'keyring.backends.Windows',
        'keyring.backends.SecretService',
        'keyring.backends.kwallet',
        'keyring.backends.chainer',
        'keyring.backends.fail',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtWebEngineCore', 'PySide6.QtMultimedia', 'PySide6.QtCharts', 'PySide6.QtSql', 'PySide6.QtPrintSupport', 'QtWebEngineCore', 'QtSql', 'QtNetwork', 'QtTest', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pixai_tagger_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons\\app_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pixai_tagger_gui',
)
