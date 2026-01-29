# /mnt/e/env/ts/model_info/src/model_info/utils/text.py
from __future__ import annotations
import re
import hashlib
from typing import Any, Tuple

SCALAR_TYPES = (type(None), bool, int, float, str)

def slugify(x: Any, max_len: int = 120) -> str:
    s = "" if x is None else str(x).strip()
    if not s:
        return "_unknown"
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:max_len]

def stable_repr(x: Any) -> str:
    s = repr(x)
    # メモリアドレス 0x... を消して差分比較しやすくする
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    return s

def object_id(x: Any) -> str:
    s = f"{type(x)}|{stable_repr(x)}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def split_module(module: str) -> Tuple[str, str, str, str]:
    parts = (module or "").split(".")
    library = parts[0] if len(parts) > 0 else ""
    namespace = parts[1] if len(parts) > 1 else ""
    submodule_head = parts[2] if len(parts) > 2 else ""
    submodule = ".".join(parts[2:]) if len(parts) > 2 else ""
    return library, namespace, submodule_head, submodule

def short_scalar(x: Any, max_len: int = 80) -> Any:
    if isinstance(x, str) and len(x) > max_len:
        return x[: max_len - 3] + "..."
    return x
