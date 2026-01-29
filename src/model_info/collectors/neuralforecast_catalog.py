# /mnt/e/env/ts/model_info/src/model_info/collectors/neuralforecast_catalog.py
from __future__ import annotations
import importlib
import pkgutil
import inspect
import re
from typing import Dict, Tuple
import pandas as pd

import neuralforecast.models as nf_models
import neuralforecast.auto as nf_auto

from model_info.utils.web_cache import fetch_and_cache
from model_info.utils.text import split_module

DEFAULT_MODELS_URL = "https://nixtlaverse.nixtla.io/neuralforecast/models.html"
DEFAULT_LLMS_URL = "https://nixtlaverse.nixtla.io/llms.txt"

def _parse_automodel_family_map(html: str) -> Dict[str, str]:
    # HTMLを粗くテキスト化
    text = re.sub(r"<[^>]+>", "\n", html)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    family = None
    fam_map: Dict[str, str] = {}

    def norm(h: str) -> str:
        h = h.lower()
        if "rnn-based" in h: return "RNN"
        if "transformer-based" in h: return "Transformer"
        if "cnn-based" in h: return "CNN"
        if "linear" in h or "mlp" in h: return "Linear/MLP"
        if "specialized" in h: return "Specialized"
        return "Other"

    for ln in lines:
        if ("RNN-Based Models" in ln or "Transformer-Based Models" in ln or "CNN-Based Models" in ln
            or "Linear and MLP Models" in ln or "Specialized Models" in ln):
            family = norm(ln)
            continue
        if family:
            names = re.findall(r"\bAuto[A-Za-z0-9_]+\b", ln)
            for n in names:
                fam_map[n] = family

    return fam_map

def _collect_models_only() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # BaseModelのimportパス差異に備えてtry
    BaseModel = None
    try:
        from neuralforecast.common._base_model import BaseModel  # type: ignore
    except Exception:
        BaseModel = None

    rows = []
    errors = []

    for modinfo in pkgutil.iter_modules(nf_models.__path__, nf_models.__name__ + "."):
        modname = modinfo.name
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            errors.append({"module": modname, "error": repr(e)})
            continue

        for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__:
                continue
            if cls_name.startswith("_"):
                continue
            if BaseModel is not None:
                try:
                    if not issubclass(cls, BaseModel):  # type: ignore[arg-type]
                        continue
                except Exception:
                    continue
            # BaseModelが取れない場合でも「モジュール内クラス」を拾う（安全側）
            doc = inspect.getdoc(cls) or ""
            first = doc.splitlines()[0] if doc else ""
            rows.append({
                "kind": "model",
                "name": cls_name,
                "module": cls.__module__,
                "qualname": f"{cls.__module__}.{cls_name}",
                "doc": first,
            })

    models_df = (pd.DataFrame(rows)
        .drop_duplicates(subset=["qualname"])
        .sort_values(["name"])
        .reset_index(drop=True)
    )
    import_errors_df = pd.DataFrame(errors, columns=["module", "error"])
    if len(import_errors_df) > 0:
        import_errors_df = import_errors_df.sort_values(["module"]).reset_index(drop=True)

    return models_df, import_errors_df

def _collect_auto_models(family_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for cls_name, cls in inspect.getmembers(nf_auto, inspect.isclass):
        if cls.__module__ != nf_auto.__name__:
            continue
        if not cls_name.startswith("Auto"):
            continue
        if cls_name.startswith("_"):
            continue

        doc = inspect.getdoc(cls) or ""
        first = doc.splitlines()[0] if doc else ""
        rows.append({
            "kind": "auto_model",
            "name": cls_name,
            "module": cls.__module__,
            "qualname": f"{cls.__module__}.{cls_name}",
            "doc": first,
            "family": family_map.get(cls_name, ""),
        })

    auto_df = (pd.DataFrame(rows)
        .drop_duplicates(subset=["qualname"])
        .sort_values(["name"])
        .reset_index(drop=True)
    )
    return auto_df

def build_neuralforecast_catalog(
    cache_dir: str,
    models_url: str = DEFAULT_MODELS_URL,
    llms_url: str = DEFAULT_LLMS_URL,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      catalog_df: models + auto_model を統合したカタログ（ツリー保存向け）
      import_errors_df: models import失敗一覧
    """
    # docs cache
    html = fetch_and_cache(models_url, f"{cache_dir}/models.html")
    _ = fetch_and_cache(llms_url, f"{cache_dir}/llms.txt")

    fam_map = _parse_automodel_family_map(html)

    models_df, import_errors_df = _collect_models_only()
    auto_df = _collect_auto_models(fam_map)

    # models側へfamilyを伝播（Auto→wrappedモデルの想定）
    # 例外だけ補正
    overrides = {
        "AutoAutoformer": "Autoformer",
        "AutoFEDformer": "FEDformer",
        "AutoiTransformer": "iTransformer",
        "AutoxLSTM": "xLSTM",
        "AutoTimeXer": "TimeXer",
    }
    model_family = {}
    for auto_name, fam in fam_map.items():
        wrapped = overrides.get(auto_name, auto_name.replace("Auto", "", 1))
        model_family[wrapped] = fam

    models_df = models_df.copy()
    models_df["family"] = models_df["name"].map(model_family).fillna("")

    catalog_df = (pd.concat([models_df, auto_df], ignore_index=True)
        .sort_values(["kind", "family", "name"])
        .reset_index(drop=True)
    )
    return catalog_df, import_errors_df
