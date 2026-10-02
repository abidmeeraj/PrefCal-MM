"""Image loading from the filesystem or (optionally) an LMDB store."""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_lmdb_env = None
_lmdb_path: Optional[str] = None


def load_image_file(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.load()
    return img


def load_image_dir(d: Path) -> List[Image.Image]:
    """All images in ``d``, sorted by filename. Missing directory -> []."""
    if not d.is_dir():
        return []
    out: List[Image.Image] = []
    for name in sorted(os.listdir(d)):
        if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
            continue
        try:
            out.append(load_image_file(d / name))
        except (OSError, IOError) as e:
            logger.warning("Could not load image %s: %s", d / name, e)
    return out


def _normalise_key(key: str) -> str:
    s = str(key).replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s.strip()


def _open_lmdb(path: str):
    """Open (and cache) a read-only LMDB environment."""
    global _lmdb_env, _lmdb_path
    resolved = str(Path(path).resolve())
    if _lmdb_env is not None and _lmdb_path == resolved:
        return _lmdb_env
    try:
        import lmdb
    except ImportError as e:
        raise ImportError("Reading images from LMDB requires `pip install lmdb`.") from e
    if not Path(resolved).exists():
        raise FileNotFoundError(f"LMDB not found at {resolved}")
    _lmdb_env = lmdb.open(resolved, readonly=True, lock=False, readahead=False, meminit=False)
    _lmdb_path = resolved
    return _lmdb_env


def load_lmdb_image(lmdb_path: str, key: str) -> Image.Image:
    """Decode the image stored under ``key`` (UTF-8 encoded path string)."""
    env = _open_lmdb(lmdb_path)
    nkey = _normalise_key(key)
    with env.begin(buffers=True) as txn:
        value = txn.get(nkey.encode("utf-8"))
        if value is None:
            raise KeyError(f"LMDB key not found: {nkey!r}")
        img = Image.open(io.BytesIO(bytes(value)))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
        return img


def load_images(
    refs: List[str],
    *,
    image_root: Optional[Path] = None,
    lmdb_path: Optional[str] = None,
    context: str = "",
) -> List[Image.Image]:
    """Load a list of image references, skipping (with a warning) any that fail.

    With ``lmdb_path`` set, references are LMDB keys; otherwise they are file
    paths, resolved against ``image_root`` when relative.
    """
    out: List[Image.Image] = []
    for ref in refs:
        try:
            if lmdb_path:
                out.append(load_lmdb_image(lmdb_path, ref))
            else:
                p = Path(ref).expanduser()
                if not p.is_absolute() and image_root is not None:
                    p = image_root / p
                out.append(load_image_file(p))
        except (KeyError, OSError, IOError) as e:
            logger.warning("Skipping image %r%s: %s", ref, f" ({context})" if context else "", e)
    return out
