"""Generic CSV dataset: score your own multimodal summaries.

One row per summary. Image columns hold a JSON (or Python) list of image
references: file paths, resolved against ``image_root`` when relative, or LMDB
keys when ``lmdb_path`` is set. Column names are configured per dataset in
``configs/config.yaml::datasets`` (see the ``custom_csv`` template).
"""
from __future__ import annotations

import ast
import csv
import json
import logging
import sys
from typing import Any, Dict, Iterator, List, Optional

from ..config import resolve_path
from .images import load_images as _load_refs

logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

_REQUIRED_KEYS = ("csv_path", "id_column", "source_text_column", "summary_column")


def parse_list_column(raw: str) -> List[str]:
    """Parse a JSON or Python list literal into ``list[str]``; '' -> []."""
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not parse image list %r: %s", raw[:120], e)
            return []
    if not isinstance(parsed, list):
        return []
    return [str(x) for x in parsed if x]


def iter_records(
    cfg: Dict[str, Any],
    *,
    load_images: bool = True,
    max_records: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield records in file order using the column mapping in ``cfg``."""
    missing = [k for k in _REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(f"CSV dataset config is missing {missing}")

    csv_file = resolve_path(cfg["csv_path"])
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    id_col = cfg["id_column"]
    src_col = cfg["source_text_column"]
    sum_col = cfg["summary_column"]
    src_img_col = cfg.get("source_images_column")
    sel_img_col = cfg.get("selected_images_column")
    sys_col = cfg.get("system_column")
    default_system = str(cfg.get("system_name") or "system")
    image_root = resolve_path(cfg["image_root"]) if cfg.get("image_root") else None
    lmdb_path = str(resolve_path(cfg["lmdb_path"])) if cfg.get("lmdb_path") else None

    with csv_file.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = [c for c in (id_col, src_col, sum_col, src_img_col, sel_img_col, sys_col) if c]
        absent = [c for c in columns if c not in (reader.fieldnames or [])]
        if absent:
            raise ValueError(f"{csv_file} is missing columns {absent}; found {reader.fieldnames}")

        n = 0
        for row in reader:
            article_id = (row.get(id_col) or "").strip()
            if not article_id:
                continue
            system_name = (row.get(sys_col) or "").strip() if sys_col else ""
            system_name = system_name or default_system

            src_refs = parse_list_column(row.get(src_img_col, "")) if src_img_col else []
            sel_refs = parse_list_column(row.get(sel_img_col, "")) if sel_img_col else []
            ctx = f"{article_id}_{system_name}"
            source_images = (
                _load_refs(src_refs, image_root=image_root, lmdb_path=lmdb_path, context=ctx)
                if load_images else []
            )
            selected_images = (
                _load_refs(sel_refs, image_root=image_root, lmdb_path=lmdb_path, context=ctx)
                if load_images else []
            )

            yield {
                "pair_id": f"{article_id}_{system_name}",
                "article_id": article_id,
                "system_name": system_name,
                "article": {
                    "article_id": article_id,
                    "source_text": row.get(src_col, "") or "",
                    "source_images": source_images,
                },
                "summary": {
                    "system_name": system_name,
                    "text": row.get(sum_col, "") or "",
                    "selected_images": selected_images,
                },
            }
            n += 1
            if max_records is not None and n >= max_records:
                return
