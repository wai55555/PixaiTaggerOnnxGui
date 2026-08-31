# -*- mode: python ; coding: utf-8 -*-

import glob
import os

# Ship every model's hand-authored model_config.json (NOT the multi-GB model.onnx files,
# which the app downloads at runtime) plus PixAI's curated tag-translation CSVs, keeping
# the models/<model_id>/ directory structure so model_registry.discover_models() finds them.
_model_datas = [
    (p, os.path.dirname(p))
    for p in glob.glob('models/*/model_config.json')
]
_model_datas += [
    (p, os.path.dirname(p))
    for p in glob.glob('models/pixai-tagger-v0.9/selected_tags*.csv')
]


a = Analysis(
    ['pixai_tagger_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('icons', 'icons'), ('lang', 'lang')] + _model_datas,
    hiddenimports=['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets'],
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
