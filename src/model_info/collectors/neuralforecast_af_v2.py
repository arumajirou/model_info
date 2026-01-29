# /mnt/e/env/ts/model_info/src/model_info/collectors/neuralforecast_af_v2.py
from __future__ import annotations
import inspect
from typing import Any, Dict, List, Tuple
import pandas as pd

import neuralforecast.auto as nf_auto

from model_info.utils.text import SCALAR_TYPES, stable_repr, object_id, short_scalar, split_module
from model_info.utils.web_cache import fetch_and_cache

DEFAULT_MODELS_URL = "https://nixtlaverse.nixtla.io/neuralforecast/models.html"
DEFAULT_LLMS_URL = "https://nixtlaverse.nixtla.io/llms.txt"

PARAM_GROUP = {
    "h": "forecasting",
    "loss": "loss",
    "valid_loss": "loss",
    "config": "search_space",
    "search_alg": "search_space",
    "num_samples": "search_space",
    "backend": "search_space",
    "callbacks": "search_space",
    "cpus": "resources",
    "gpus": "resources",
    "refit_with_val": "workflow",
    "verbose": "workflow",
    "alias": "workflow",
    "S": "hierarchical",
    "cls_model": "hierarchical",
    "reconciliation": "hierarchical",
    "n_series": "data",
}

def classify_param(pname: str) -> str:
    return PARAM_GROUP.get(pname, "other")

def flatten_config(auto_name: str, obj: Any, key_path: str, config_rows: List[Dict[str, Any]], obj_rows: Dict[str, Dict[str, str]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{key_path}.{k}" if key_path else str(k)
            flatten_config(auto_name, v, kp, config_rows, obj_rows)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            kp = f"{key_path}.{i}" if key_path else str(i)
            flatten_config(auto_name, v, kp, config_rows, obj_rows)
        return

    if isinstance(obj, SCALAR_TYPES):
        config_rows.append({
            "auto_name": auto_name,
            "key_path": key_path,
            "value_kind": "scalar",
            "value_scalar": short_scalar(obj),
            "value_obj_id": "",
        })
        return

    oid = object_id(obj)
    if oid not in obj_rows:
        obj_rows[oid] = {"obj_id": oid, "py_type": str(type(obj)), "repr": stable_repr(obj)}
    config_rows.append({
        "auto_name": auto_name,
        "key_path": key_path,
        "value_kind": "object",
        "value_scalar": "",
        "value_obj_id": oid,
    })

def try_get_default_config(cls: Any) -> Any:
    if hasattr(cls, "default_config"):
        try:
            cfg = getattr(cls, "default_config")
            if callable(cfg):
                cfg = cfg()
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
    if hasattr(cls, "get_default_config"):
        try:
            fn = getattr(cls, "get_default_config")
            if callable(fn):
                cfg = fn()
                if isinstance(cfg, dict):
                    return cfg
        except Exception:
            pass
    return None

def build_neuralforecast_af_v2(
    cache_dir: str,
    models_url: str = DEFAULT_MODELS_URL,
    llms_url: str = DEFAULT_LLMS_URL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Automatic Forecasting v2 (正規化スキーマ)
      A) af_models_df
      B) af_params_df
      C) af_model_params_df
      D) af_config_entries_df
      E) af_objects_df
    """
    # docs cache（将来の拡張・差分検知用。現段階では保存するだけ）
    _ = fetch_and_cache(models_url, f"{cache_dir}/models.html")
    _ = fetch_and_cache(llms_url, f"{cache_dir}/llms.txt")

    # Autoクラス一覧
    auto_classes = []
    for name, cls in inspect.getmembers(nf_auto, inspect.isclass):
        if cls.__module__ != nf_auto.__name__:
            continue
        if not name.startswith("Auto"):
            continue
        if name.startswith("_"):
            continue
        auto_classes.append((name, cls))

    models_rows = []
    params_master: Dict[str, Dict[str, str]] = {}
    model_params_rows = []
    config_rows = []
    obj_store: Dict[str, Dict[str, str]] = {}

    for auto_name, cls in sorted(auto_classes, key=lambda x: x[0]):
        module = cls.__module__
        library, namespace, submodule_head, submodule = split_module(module)

        doc = inspect.getdoc(cls) or ""
        doc1 = doc.splitlines()[0] if doc else ""

        models_rows.append({
            "auto_name": auto_name,
            "module": module,
            "library": library,
            "namespace": namespace,
            "submodule": submodule,
            "doc": doc1,
            "has_default_config_attr": hasattr(cls, "default_config"),
            "has_get_default_config": hasattr(cls, "get_default_config"),
            "family": "",
        })

        # signature解析（引数テーブルを正規化）
        try:
            sig = inspect.signature(cls)
        except Exception:
            sig = None

        if sig is not None:
            for pname, p in sig.parameters.items():
                if pname in ("self", "args", "kwargs"):
                    continue
                required = (p.default is inspect._empty)

                default_kind = "empty"
                default_scalar = ""
                default_obj_id = ""

                if not required:
                    dv = p.default
                    if isinstance(dv, SCALAR_TYPES):
                        default_kind = "scalar"
                        default_scalar = short_scalar(dv)
                    else:
                        default_kind = "object"
                        oid = object_id(dv)
                        default_obj_id = oid
                        if oid not in obj_store:
                            obj_store[oid] = {"obj_id": oid, "py_type": str(type(dv)), "repr": stable_repr(dv)}

                anno = "" if p.annotation is inspect._empty else (getattr(p.annotation, "__name__", None) or str(p.annotation))
                if pname not in params_master:
                    params_master[pname] = {
                        "param": pname,
                        "param_group": classify_param(pname),
                        "annotation": anno,
                    }

                model_params_rows.append({
                    "auto_name": auto_name,
                    "param": pname,
                    "required": required,
                    "default_kind": default_kind,
                    "default_scalar": default_scalar,
                    "default_obj_id": default_obj_id,
                    "kind": str(p.kind),
                })

        # default_config の正規化（dict/listは別テーブルへ）
        cfg = try_get_default_config(cls)
        if isinstance(cfg, dict):
            flatten_config(auto_name, cfg, "default_config", config_rows, obj_store)

    af_models_df = pd.DataFrame(models_rows).sort_values(["auto_name"]).reset_index(drop=True)
    af_params_df = pd.DataFrame(list(params_master.values())).sort_values(["param_group", "param"]).reset_index(drop=True)
    af_model_params_df = pd.DataFrame(model_params_rows).sort_values(["auto_name", "param"]).reset_index(drop=True)
    af_config_entries_df = pd.DataFrame(config_rows).sort_values(["auto_name", "key_path"]).reset_index(drop=True)
    af_objects_df = pd.DataFrame(list(obj_store.values())).sort_values(["obj_id"]).reset_index(drop=True)

    return af_models_df, af_params_df, af_model_params_df, af_config_entries_df, af_objects_df
