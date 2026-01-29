# /mnt/e/env/ts/model_info/src/model_info/io/tree_save.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd

from model_info.utils.text import slugify, split_module

def ensure_module_splits(df: pd.DataFrame, module_col: str = "module") -> pd.DataFrame:
    need = {"library", "namespace", "submodule_head", "submodule"}
    if need.issubset(df.columns):
        return df

    out = df.copy()
    cols = out[module_col].apply(lambda m: pd.Series(split_module(m)))
    cols.columns = ["library", "namespace", "submodule_head", "submodule"]
    out = pd.concat([out, cols], axis=1)
    return out

def save_catalog_tree(
    df: pd.DataFrame,
    base_dir: str,
    catalog_name: str,
    group_cols: Tuple[str, ...] = ("library", "namespace", "kind", "family"),
    encoding: str = "utf-8-sig",
) -> Tuple[str, str]:
    """
    1) 全体: <base_dir>/<catalog_name>/all.csv
    2) 分割: <base_dir>/<catalog_name>/<library>/<namespace>/<kind>/<family>/catalog.csv
    """
    base = Path(base_dir).resolve()
    out_root = (base / catalog_name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    df2 = ensure_module_splits(df)

    all_path = out_root / "all.csv"
    df2.to_csv(str(all_path), index=False, encoding=encoding)

    cols = [c for c in group_cols if c in df2.columns]
    if cols:
        for key, g in df2.groupby(cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            subdir = out_root
            for v in key_tuple:
                subdir = subdir / slugify(v)
            subdir.mkdir(parents=True, exist_ok=True)
            g.to_csv(str(subdir / "catalog.csv"), index=False, encoding=encoding)

    return str(all_path), str(out_root)
