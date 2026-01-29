# /mnt/e/env/ts/model_info/src/model_info/io/save_csv.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd

def save_df_csv(df: pd.DataFrame, path: str, encoding: str = "utf-8-sig") -> str:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(p), index=False, encoding=encoding)
    return str(p)

def save_many_csv(dfs: Dict[str, pd.DataFrame], out_dir: str, encoding: str = "utf-8-sig") -> Dict[str, str]:
    base = Path(out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in dfs.items():
        out[name] = save_df_csv(df, str(base / f"{name}.csv"), encoding=encoding)
    return out
