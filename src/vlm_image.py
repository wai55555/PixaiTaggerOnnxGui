"""VLM 送信用の画像前処理（260901_VLM_spec.md 6章 / design.md 4.7節）。

すべての内蔵接続へ「できるだけ同じ画像データ」を送る。切り抜きはしない。

  1. EXIF 回転を適用
  2. RGB へ変換（透過は指定色で平坦化）
  3. アスペクト比維持で長辺を制限
  4. JPEG / PNG へエンコード
  5. MIME 確定
  6. Base64 / Data URL 化
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImagePreprocessConfig:
    max_long_edge: int = 1536
    fmt: str = "auto"            # auto | jpeg | png
    jpeg_quality: int = 90
    flatten_rgba_color: tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64}"


def _target_size(w: int, h: int, max_long_edge: int) -> tuple[int, int]:
    longest = max(w, h)
    if longest <= max_long_edge or longest == 0:
        return w, h
    scale = max_long_edge / longest
    return max(1, round(w * scale)), max(1, round(h * scale))


def _choose_format(cfg: ImagePreprocessConfig, had_alpha: bool) -> str:
    if cfg.fmt in ("jpeg", "png"):
        return cfg.fmt
    # auto: 透過があった画像でも背景を平坦化済みなので JPEG でよい。写真主体の
    # データセット用途では JPEG が無難（サイズが小さくトークン消費も抑えられる）。
    return "jpeg"


def prepare_image(source: Path | bytes | Image.Image, cfg: ImagePreprocessConfig | None = None) -> PreparedImage:
    """パス / バイト列 / PIL.Image を受け取り、送信用に整えた PreparedImage を返す。"""
    cfg = cfg or ImagePreprocessConfig()

    # opened だけが「こちらが開いたファイルハンドル」。img は EXIF回転 / convert / resize で
    # 何度も貼り替わるので、finally で img.close() すると元のハンドルを閉じられず fd を
    # リークする（一括処理で致命的）。開いた本体を別名で握って確実に閉じる。
    opened: Image.Image | None = None
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = opened = Image.open(io.BytesIO(bytes(source)))
    else:
        img = opened = Image.open(source)

    try:
        # 1. EXIF 回転
        img = ImageOps.exif_transpose(img)

        had_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)

        # 2. RGB へ（透過は指定色で平坦化）
        if had_alpha:
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, cfg.flatten_rgba_color)
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 3. 長辺制限（拡大はしない）
        tw, th = _target_size(img.width, img.height, cfg.max_long_edge)
        if (tw, th) != (img.width, img.height):
            img = img.resize((tw, th), Image.Resampling.LANCZOS)

        # 4/5. エンコード + MIME
        fmt = _choose_format(cfg, had_alpha)
        buf = io.BytesIO()
        if fmt == "jpeg":
            img.save(buf, format="JPEG", quality=cfg.jpeg_quality, optimize=True)
            mime = "image/jpeg"
        else:
            img.save(buf, format="PNG", optimize=True)
            mime = "image/png"
        return PreparedImage(data=buf.getvalue(), mime_type=mime)
    finally:
        if opened is not None:
            opened.close()
