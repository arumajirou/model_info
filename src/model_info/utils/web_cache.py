# /mnt/e/env/ts/model_info/src/model_info/utils/web_cache.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import requests

def fetch_and_cache(url: str, cache_path: str, timeout: int = 30, force: bool = False) -> str:
    p = Path(cache_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not force:
        return p.read_text(encoding="utf-8", errors="ignore")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    text = r.text
    p.write_text(text, encoding="utf-8")
    return text
