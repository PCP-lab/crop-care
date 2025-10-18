# ~/plant_disease/app/main.py
import os, io, re, json, math, base64
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
from PIL import Image

import torch
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ========================
# Paths, device, defaults
# ========================
HOME   = "/home/myid/bp67339"
ROOT   = f"{HOME}/plant_disease"
DATA   = f"{ROOT}/data"
MODELS = f"{ROOT}/models"

YOLO_W = f"{MODELS}/yolov8_seg_best.pt"

TEXT_DIR_CANDIDATES = [
    f"{MODELS}/deberta_v3_base_textclf_phase2",
    f"{MODELS}/deberta_v3_base_textclf",
]
TEXT_DIR = next((p for p in TEXT_DIR_CANDIDATES if os.path.isdir(p)), None)

DISEASE_JSON       = f"{MODELS}/disease_texts.json"
WEATHER_KB_PATH    = f"{DATA}/kb_leaf_spots.json"  # used only by /predict-forecast
MANAGEMENT_KB_PATH = f"{DATA}/management_kb.json"  # tips for rank-1 disease

# Fusion weights (img/text)
W_TEXT_DEFAULT = 0.65
W_IMG_DEFAULT  = 0.35

# Weather fusion weight kept for /predict-forecast
W_WX_DEFAULT   = 0.40
WX_FLOOR       = 0.05
REGION_MODE    = "mul"    # "mul": multiply (1+bump); "add": add bump

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# FastAPI app + CORS
# ========================
app = FastAPI(title="UGA Leaf Disease API (img+text, forecast optional)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def softmax(x, tau=1.0, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = x / max(tau, 1e-8)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + eps)

# ========================
# CLIP (open_clip)
# ========================
import open_clip
CLIP_CANDIDATES = [
    ("ViT-L-14-336", "openai"),
    ("ViT-L-14",     "laion2b_s32b_b82k"),
]
clip_model = None
preprocess = None
clip_tokenizer  = None
_loaded_name, _loaded_tag = None, None

for model_name, pretrained in CLIP_CANDIDATES:
    try:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        clip_tokenizer = open_clip.get_tokenizer(model_name)
        if device == "cuda":
            clip_model = clip_model.half()
        _loaded_name, _loaded_tag = model_name, pretrained
        print(f"[CLIP] Loaded {model_name}/{pretrained} on {device}")
        break
    except Exception as e:
        print(f"[CLIP] Failed {model_name}/{pretrained}: {e}")
if clip_model is None:
    raise RuntimeError("Could not load any CLIP model.")

CLIP_TEMPLATES = [
    "a photo of a leaf with {}",
    "a plant leaf showing {}",
    "a close-up of a leaf with {}",
    "a leaf exhibiting {}",
    "a macro photo of leaf {}",
]
_CLIP_TEXT_CACHE: Dict[Tuple[str, str, str], torch.Tensor] = {}

@torch.no_grad()
def encode_phrases_cached(phrases: List[str]) -> torch.Tensor:
    out = []
    for phrase in phrases:
        key = (_loaded_name, _loaded_tag, phrase)
        if key not in _CLIP_TEXT_CACHE:
            feats = []
            for tmpl in CLIP_TEMPLATES:
                s = tmpl.format(phrase)
                toks = clip_tokenizer([s]).to(device)
                f = clip_model.encode_text(toks)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f)
            _CLIP_TEXT_CACHE[key] = torch.stack(feats, dim=0).mean(dim=0)
        out.append(_CLIP_TEXT_CACHE[key])
    return torch.cat(out, dim=0)

@torch.no_grad()
def clip_rank(image_pil: Image.Image, phrases: List[str], topk=5):
    img = preprocess(image_pil.convert("RGB")).unsqueeze(0).to(device)
    if device == "cuda" and next(clip_model.parameters()).dtype == torch.float16:
        img = img.half()
    img_f = clip_model.encode_image(img)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = encode_phrases_cached(phrases).to(img_f.device)
    if device == "cuda" and next(clip_model.parameters()).dtype == torch.float16:
        txt_f = txt_f.half()
    logits = 100.0 * img_f @ txt_f.T
    probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    idx    = probs.argsort()[::-1][:topk]
    return [(phrases[i], float(probs[i])) for i in idx], probs

# ========================
# YOLOv8-seg (Ultralytics)
# ========================
from ultralytics import YOLO
yolo_seg = YOLO(YOLO_W)
print("[YOLO] Loaded:", YOLO_W)

def segment_and_overlay_rgb(image_pil: Image.Image) -> np.ndarray:
    r = yolo_seg.predict(image_pil, imgsz=1280, conf=0.20, verbose=False)[0]
    bgr = r.plot()                 # ndarray in BGR
    return bgr[..., ::-1].copy()   # BGR -> RGB

def png_base64_from_rgb(rgb_np: np.ndarray) -> str:
    img = Image.fromarray(rgb_np)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _class_ids_by_name(names_map, wanted: set[str]) -> list[int]:
    return [i for i, nm in names_map.items() if nm.lower() in wanted]

def analyze_segmentation(image_pil: Image.Image):
    """
    Return overlay + coverage dict: {'leaf_pct': float|None, 'spot_pct': float|None}
    Percentages are relative to full image area (0..100).
    """
    r = yolo_seg.predict(image_pil, imgsz=1280, conf=0.20, verbose=False)[0]
    bgr = r.plot()
    overlay_rgb = bgr[..., ::-1].copy()

    H, W = overlay_rgb.shape[:2]
    img_area = float(H * W)

    masks = getattr(r, "masks", None)
    boxes = getattr(r, "boxes", None)
    names = getattr(r, "names", {})

    if masks is None or boxes is None or img_area <= 0:
        return overlay_rgb, {"leaf_pct": None, "spot_pct": None}

    cls = boxes.cls.cpu().numpy().astype(int)
    mdata = masks.data.cpu().numpy()  # [N, H, W]

    leaf_ids = set(_class_ids_by_name(names, {"leaf", "leaves"}))
    spot_ids = set(_class_ids_by_name(names, {"spot", "leaf_spot", "leaf-spot", "lesion", "disease_spot"}))

    leaf_mask_union = None
    spot_mask_union = None

    for i, cid in enumerate(cls):
        m = (mdata[i] > 0.5).astype(np.uint8)
        if cid in leaf_ids:
            leaf_mask_union = m if leaf_mask_union is None else (leaf_mask_union | m)
        if cid in spot_ids:
            spot_mask_union = m if spot_mask_union is None else (spot_mask_union | m)

    leaf_pct = None
    spot_pct = None
    if leaf_mask_union is not None:
        leaf_area = float(leaf_mask_union.sum())
        leaf_pct = 100.0 * (leaf_area / img_area)
    if spot_mask_union is not None:
        spot_area = float(spot_mask_union.sum())
        spot_pct = 100.0 * (spot_area / img_area)

    return overlay_rgb, {"leaf_pct": leaf_pct, "spot_pct": spot_pct}

# ========================
# DeBERTa text classifier
# ========================
from transformers import AutoTokenizer, AutoModelForSequenceClassification

assert TEXT_DIR, f"Could not find a text model. Checked: {TEXT_DIR_CANDIDATES}"
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_DIR, local_files_only=True, use_fast=False)
text_model     = AutoModelForSequenceClassification.from_pretrained(
    TEXT_DIR, local_files_only=True, use_safetensors=True
).to(device).eval()

LABELS: List[str] = []
labels_json_fp = os.path.join(TEXT_DIR, "labels.json")
if os.path.isfile(labels_json_fp):
    try:
        with open(labels_json_fp, "r") as f:
            LABELS = json.load(f)["labels"]
    except Exception as e:
        print(f"[WARN] Failed to read labels.json: {e}")
if not LABELS:
    id2label_cfg = getattr(text_model.config, "id2label", {}) or {}
    LABELS = [id2label_cfg[i] for i in range(len(id2label_cfg))] if id2label_cfg else []
id2label = {i: l for i, l in enumerate(LABELS)}

MAX_LEN = 256

@torch.no_grad()
def predict_text_disease(text: str) -> np.ndarray:
    enc = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(device)
    logits = text_model(**enc).logits
    probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs  # LABELS order

# ========================
# Disease bank for CLIP phrases
# ========================
if os.path.isfile(DISEASE_JSON):
    try:
        disease_texts = json.load(open(DISEASE_JSON))["disease_texts"]
        print(f"[BANK] Loaded {len(disease_texts)} disease names from disease_texts.json")
    except Exception as e:
        print(f"[BANK] Failed disease_texts.json: {e}")
        disease_texts = LABELS[:] if LABELS else []
else:
    disease_texts = LABELS[:] if LABELS else []
print(f"[BANK] {len(disease_texts)} diseases in CLIP bank")

# ========================
# Unified disease universe (labels ∪ clip bank)
# ========================
DISEASE_UNIVERSE = sorted(set(disease_texts) | set(LABELS), key=lambda s: s.lower())
U = len(DISEASE_UNIVERSE)
name_to_uix = {_norm(n): i for i, n in enumerate(DISEASE_UNIVERSE)}

def map_vector(names: List[str], probs: np.ndarray) -> np.ndarray:
    vec = np.zeros(U, dtype=np.float64)
    for n, p in zip(names, probs):
        i = name_to_uix.get(_norm(n))
        if i is not None:
            vec[i] = float(p)
    s = vec.sum()
    return vec / (s + 1e-12) if s > 0 else np.ones(U)/U

def remap_text_probs_to_universe(p_vec: np.ndarray) -> np.ndarray:
    return map_vector(LABELS, p_vec)

def image_probs_to_universe(p_vec_clip_order: np.ndarray) -> np.ndarray:
    return map_vector(disease_texts, p_vec_clip_order)

@torch.no_grad()
def image_probs_from_clip_universe(cutouts, tau_img=0.9) -> np.ndarray:
    per_leaf = []
    for im in cutouts:
        _, full = clip_rank(im, disease_texts, topk=len(disease_texts))
        per_leaf.append(full)
    agg = np.maximum.reduce(per_leaf) if per_leaf else np.ones(len(disease_texts))/len(disease_texts)
    p_clip = softmax(agg, tau=tau_img)
    return image_probs_to_universe(p_clip)

def combine_product_of_experts(parts: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """
    parts: list of (prob_vector_in_universe, weight)
    """
    eps = 1e-12
    logp = None
    for p, w in parts:
        if p is None or w is None or w == 0:
            continue
        p = np.asarray(p, dtype=np.float64)
        p = p / (p.sum() + eps)
        term = float(w) * np.log(p + eps)
        logp = term if logp is None else (logp + term)
    if logp is None:
        return np.ones(U)/U
    return softmax(logp, tau=1.0)

# ========================
# Weather forecast: fetch + features (used only by /predict-forecast)
# ========================
def fetch_open_meteo_forecast(
    lat: float,
    lon: float,
    timezone_str: str = "UTC",
    forecast_days: int = 16,
    hourly_vars: Optional[List[str]] = None,
    daily_vars: Optional[List[str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    base = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = hourly_vars or [
        "temperature_2m", "relative_humidity_2m", "precipitation", "cloudcover"
    ]
    daily_vars = daily_vars or [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum"
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone_str,
        "forecast_days": int(min(16, max(1, forecast_days))),
        "past_days": 0,
        "hourly": ",".join(hourly_vars),
        "daily": ",".join(daily_vars),
    }
    r = requests.get(base, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _dew_point_c(T_c: float, RH: float) -> float:
    a, b = 17.27, 237.3
    RHc = max(1e-6, min(1.0, RH/100.0))
    gamma = (a*T_c)/(b+T_c) + math.log(RHc)
    return (b*gamma)/(a - gamma)

def _vpd_kpa(T_c: float, RH: float) -> float:
    es = 0.6108 * math.exp((17.27*T_c)/(T_c+237.3))
    ea = es * (max(0.0, min(100.0, RH))/100.0)
    return max(0.0, es - ea)

def _compute_leaf_wetness_hours(hourly_df: pd.DataFrame) -> int:
    if hourly_df.empty:
        return 0
    Td = hourly_df.apply(lambda r: _dew_point_c(r["temperature_2m"], r["relative_humidity_2m"]), axis=1)
    return int(((hourly_df["relative_humidity_2m"] >= 90) | ((hourly_df["temperature_2m"] - Td) <= 2.0)).sum())

def build_forecast_tables(raw_json: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_df = pd.DataFrame(raw_json.get("hourly", {}))
    daily_df  = pd.DataFrame(raw_json.get("daily",  {}))
    if not hourly_df.empty:
        hourly_df["time"] = pd.to_datetime(hourly_df["time"], utc=True)
        hourly_df = hourly_df.set_index("time").sort_index()
    if not daily_df.empty:
        daily_df["time"] = pd.to_datetime(daily_df["time"], utc=True)
        daily_df = daily_df.set_index("time").sort_index()
    return hourly_df, daily_df

def compute_forecast_features(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, window_days: int = 7, **kwargs) -> Dict[str, Any]:
    """
    Compute forecast features over the next `window_days` days.
    NOTE: Field names keep the *_7 suffix for backward-compat even when window_days != 7.
    """
    if hourly_df.empty and daily_df.empty:
        return {
            "window_end_date": None,
            "T_mean_7": float("nan"), "T_min_7": float("nan"), "T_max_7": float("nan"),
            "RH_mean_7": float("nan"), "RH_max_7": float("nan"),
            "Rain_sum_7": float("nan"), "Rain_days_7": 0, "Rain_hours_7": 0,
            "RH_hours_ge90_7": 0, "LWD_hours_7": 0,
            "VPD_mean_7": float("nan"), "VPD_max_7": float("nan"),
            "GDD_10_14": float("nan"), "Humid_streak_10": 0,
        }

    # anchor at the start of series
    start_ts = hourly_df.index.min() if not hourly_df.empty else daily_df.index.min()
    endW_ts  = start_ts + pd.Timedelta(days=int(window_days))
    end14_ts = start_ts + pd.Timedelta(days=14)

    hW  = hourly_df.loc[(hourly_df.index >= start_ts) & (hourly_df.index < endW_ts)] if not hourly_df.empty else pd.DataFrame()
    h14 = hourly_df.loc[(hourly_df.index >= start_ts) & (hourly_df.index < end14_ts)] if not hourly_df.empty else pd.DataFrame()
    dW  = daily_df.iloc[:int(window_days)] if not daily_df.empty else pd.DataFrame()
    d14 = daily_df.iloc[:14] if not daily_df.empty else pd.DataFrame()

    # Temps
    if not dW.empty:
        daily_mean_W = (dW.get("temperature_2m_max", pd.Series(dtype=float)) + dW.get("temperature_2m_min", pd.Series(dtype=float))) / 2.0
        T_mean_7 = float(daily_mean_W.mean())
        T_min_7  = float(dW.get("temperature_2m_min", pd.Series(dtype=float)).mean())
        T_max_7  = float(dW.get("temperature_2m_max", pd.Series(dtype=float)).mean())
    else:
        T_mean_7 = T_min_7 = T_max_7 = float("nan")

    # RH / Wetness / Rain
    if not hW.empty:
        RH_mean_7 = float(hW["relative_humidity_2m"].mean())
        RH_max_7  = float(hW["relative_humidity_2m"].max())
        RH_hours_ge90_7 = int((hW["relative_humidity_2m"] >= 90).sum())
        # leaf wetness proxy
        TdW = hW.apply(lambda r: _dew_point_c(r["temperature_2m"], r["relative_humidity_2m"]), axis=1)
        LWD_hours_7 = int(((hW["relative_humidity_2m"] >= 90) | ((hW["temperature_2m"] - TdW) <= 2.0)).sum())
        Rain_hours_7 = int((hW.get("precipitation", pd.Series(dtype=float)) > 0.1).sum())
        # VPD
        vpd_series = hW.apply(lambda r: _vpd_kpa(r["temperature_2m"], r["relative_humidity_2m"]), axis=1)
        VPD_mean_7 = float(vpd_series.mean())
        VPD_max_7  = float(vpd_series.max())
    else:
        RH_mean_7 = RH_max_7 = VPD_mean_7 = VPD_max_7 = float("nan")
        RH_hours_ge90_7 = LWD_hours_7 = Rain_hours_7 = 0

    if not dW.empty:
        Rain_sum_7  = float(dW.get("precipitation_sum", pd.Series(dtype=float)).sum()) if "precipitation_sum" in dW else float("nan")
        Rain_days_7 = int((dW.get("precipitation_sum", pd.Series(dtype=float)) > 0.1).sum()) if "precipitation_sum" in dW else 0
    else:
        Rain_sum_7 = float("nan")
        Rain_days_7 = 0

    # 14-day features used by downstream rules
    if not d14.empty:
        daily_mean_14 = (d14.get("temperature_2m_max", pd.Series(dtype=float)) + d14.get("temperature_2m_min", pd.Series(dtype=float))) / 2.0
        GDD_10_14 = float(((daily_mean_14 - 10.0).clip(lower=0.0)).sum())
    else:
        GDD_10_14 = float("nan")

    if not h14.empty:
        rh_daily_mean = h14["relative_humidity_2m"].groupby(h14.index.date).mean()
        rh_daily_10 = rh_daily_mean.iloc[:10] if len(rh_daily_mean) > 10 else rh_daily_mean
        max_streak = 0; cur = 0
        for v in rh_daily_10.values:
            if v >= 85: cur += 1; max_streak = max(max_streak, cur)
            else: cur = 0
        Humid_streak_10 = int(max_streak)
    else:
        Humid_streak_10 = 0

    return {
        "window_end_date": str((start_ts + pd.Timedelta(days=int(window_days))).date()),
        "T_mean_7": T_mean_7, "T_min_7": T_min_7, "T_max_7": T_max_7,
        "RH_mean_7": RH_mean_7, "RH_max_7": RH_max_7,
        "Rain_sum_7": Rain_sum_7, "Rain_days_7": Rain_days_7, "Rain_hours_7": Rain_hours_7,
        "RH_hours_ge90_7": RH_hours_ge90_7, "LWD_hours_7": LWD_hours_7,
        "VPD_mean_7": VPD_mean_7, "VPD_max_7": VPD_max_7,
        "GDD_10_14": GDD_10_14, "Humid_streak_10": Humid_streak_10,
    }

# ── Per-day feature builder & risk helpers (NEW) ───────────────────────────────
def _compute_daily_basic_features(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build a list of per-day feature dicts for the next 7 days:
    RH_mean_1, RH_max_1, Rain_sum_1, Rain_hours_1, LWD_hours_1, T_max_1, VPD_mean_1, date
    """
    out: List[Dict[str, Any]] = []
    if daily_df.empty:
        return out

    # First 7 forecast dates
    days = daily_df.index[:7]
    for dt in days:
        day_end = dt + pd.Timedelta(days=1)
        h = hourly_df.loc[(hourly_df.index >= dt) & (hourly_df.index < day_end)] if not hourly_df.empty else pd.DataFrame()

        if not h.empty:
            Td = h.apply(lambda r: _dew_point_c(r["temperature_2m"], r["relative_humidity_2m"]), axis=1)
            LWD_hours_1  = int(((h["relative_humidity_2m"] >= 90) | ((h["temperature_2m"] - Td) <= 2.0)).sum())
            RH_mean_1    = float(h["relative_humidity_2m"].mean())
            RH_max_1     = float(h["relative_humidity_2m"].max())
            Rain_hours_1 = int((h.get("precipitation", pd.Series(dtype=float)) > 0.1).sum())
            VPD_mean_1   = float(h.apply(lambda r: _vpd_kpa(r["temperature_2m"], r["relative_humidity_2m"]), axis=1).mean())
        else:
            LWD_hours_1 = 0
            RH_mean_1 = RH_max_1 = VPD_mean_1 = float("nan")
            Rain_hours_1 = 0

        # Daily aggregates
        row = daily_df.loc[dt]
        T_max_1    = float(row.get("temperature_2m_max", float("nan")))
        T_min_1    = float(row.get("temperature_2m_min", float("nan")))
        Rain_sum_1 = float(row.get("precipitation_sum", float("nan")))

        out.append({
            "date": str(pd.to_datetime(dt).date()),
            "RH_mean_1": RH_mean_1, "RH_max_1": RH_max_1,
            "Rain_sum_1": Rain_sum_1, "Rain_hours_1": Rain_hours_1,
            "LWD_hours_1": LWD_hours_1,
            "VPD_mean_1": VPD_mean_1,
            "T_max_1": T_max_1, "T_min_1": T_min_1,
        })
    return out

def _weather_supports_day(disease: str, f: Dict[str, Any], window_days: Optional[int] = None, **kwargs) -> bool:
    """Reuse your weekly heuristics but day-by-day."""
    d = disease.lower()
    RH = f.get("RH_mean_1")
    Rain_hours = float(f.get("Rain_hours_1", 0) or 0)
    LWD = float(f.get("LWD_hours_1", 0) or 0)
    VPD = f.get("VPD_mean_1")
    T_max = f.get("T_max_1")

    # guard NaNs
    rh_ok   = (RH is not None)  and (not math.isnan(RH))
    vpd_ok  = (VPD is not None) and (not math.isnan(VPD))
    tmax_ok = (T_max is not None) and (not math.isnan(T_max))

    if "anthracnose" in d or "leaf spot" in d:
        return (rh_ok and RH >= 80.0) and (Rain_hours >= 3 or LWD >= 2)
    if "scorch" in d:
        return (vpd_ok and VPD >= 1.5) and (tmax_ok and T_max >= 30.0) and (Rain_hours <= 1)
    return False

def _bucket(prob: float, supported: bool) -> str:
    """Same style as your weekly bucket, slightly tuned for per-day."""
    if (prob >= 0.55 and supported) or (prob >= 0.65):
        return "high"
    if prob >= 0.35 or supported:
        return "medium"
    return "low"
_SEV_RANK = {"low": 0, "medium": 1, "high": 2}
def _severity_rank(r: str) -> int:
    return _SEV_RANK.get((r or "").lower(), 0)

# ========================
# KB loading + scoring (rules + region priors) for /predict-forecast only
# ========================
_KB = None
_KB_BY_NAME = {}

def load_weather_kb(fp: str) -> dict:
    global _KB, _KB_BY_NAME
    with open(fp, "r") as f:
        _KB = json.load(f)
    _KB_BY_NAME = {_norm(d["name"]): d for d in _KB.get("diseases", [])}
    return _KB

def _safe_eval_rule(expr: str, feats: dict) -> bool:
    allowed = set(list("0123456789.+-*/()<>=! &|") + list("_") +
                  list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    s = expr.replace("&&"," and ").replace("||"," or ")
    if any(ch not in allowed for ch in s):
        return False
    for k, v in feats.items():
        try:
            fv = float(v)
            s = re.sub(rf"\b{k}\b", str(fv), s)
        except Exception:
            pass
    try:
        return bool(eval(s, {"__builtins__": {}}, {}))
    except Exception:
        return False

def score_weather_for_disease(d_entry: dict, feats: dict) -> float:
    sc = 0.0
    for r in d_entry.get("scoring", {}).get("rules", []):
        cond, add = r.get("if",""), float(r.get("add",0.0))
        if cond and _safe_eval_rule(cond, feats):
            sc += add
    cap = float(d_entry.get("scoring", {}).get("cap", 1.0))
    sc = max(0.0, min(cap, sc))
    return sc

def apply_region_bump(score: float, d_entry: dict, state_code: Optional[str]) -> float:
    if not state_code:
        return score
    priors = d_entry.get("region_priors", {}).get("by_state", {})
    bump = float(priors.get(state_code.upper(), priors.get("*", 0.0)) or 0.0)
    if REGION_MODE == "mul":
        return score * (1.0 + bump)
    else:
        return max(0.0, score + bump)

def build_weather_prior_universe_from_forecast(
    feats: dict,
    kb_path: str = WEATHER_KB_PATH,
    state_code: Optional[str] = None,
    floor: float = WX_FLOOR,
    window_days: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    global _KB, _KB_BY_NAME
    if _KB is None:
        if not os.path.isfile(kb_path):
            return np.ones(U, dtype=np.float64) / U
        load_weather_kb(kb_path)

    pw = np.zeros(U, dtype=np.float64)
    for i, name in enumerate(DISEASE_UNIVERSE):
        d = _KB_BY_NAME.get(_norm(name))
        if d is None:
            pw[i] = floor
            continue
        sc = score_weather_for_disease(d, feats)
        sc = apply_region_bump(sc, d, state_code)
        pw[i] = max(floor, sc)
    s = pw.sum()
    return pw / (s + 1e-12) if s > 0 else np.ones(U)/U

# ========================
# Management KB (return tips for top-1 disease+risk)
# ========================
_MAN = None
_MAN_BY_NAME = {}

def load_management_kb(fp: str) -> dict:
    global _MAN, _MAN_BY_NAME
    if not os.path.isfile(fp):
        _MAN = {"default_safety": {}, "diseases": []}
        _MAN_BY_NAME = {}
        return _MAN
    with open(fp, "r") as f:
        _MAN = json.load(f)
    _MAN_BY_NAME = {_norm(d["name"]): d for d in _MAN.get("diseases", [])}
    return _MAN

def get_management_for(disease_label: str, risk: str) -> dict:
    """
    Returns {"tips": [...], "safety": {...}} or {} if not found.
    """
    global _MAN, _MAN_BY_NAME
    if _MAN is None:
        load_management_kb(MANAGEMENT_KB_PATH)
    d = _MAN_BY_NAME.get(_norm(disease_label))
    if not d:
        return {}
    tips = (d.get("risk_actions", {}).get(risk.lower() if risk else "", {}) or {}).get("tips", [])
    safety = _MAN.get("default_safety", {})
    out = {}
    if tips:
        out["tips"] = tips
    if safety:
        out["safety"] = safety
    return out

# ========================
# YOLO → cutouts (RGBA)
# ========================
def segment_leaf_cutouts(image_pil: Image.Image):
    r = yolo_seg.predict(image_pil, imgsz=1280, conf=0.20, verbose=False)[0]
    masks = getattr(r, "masks", None)
    boxes = getattr(r, "boxes", None)
    if masks is None or boxes is None:
        return [image_pil.convert("RGBA")]

    names = r.names
    cls = boxes.cls.cpu().numpy().astype(int)
    leaf_ids = [i for i, k in names.items() if k.lower() in ("leaf", "leaves")]

    cutouts = []
    mdata = masks.data.cpu().numpy()  # [N, H, W]
    for i, cid in enumerate(cls):
        if cid not in leaf_ids:
            continue
        m = mdata[i]
        ys, xs = np.where(m > 0.5)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        crop = image_pil.crop((x1, y1, x2, y2))
        m_crop  = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
        mask_pil = Image.fromarray(m_crop, mode="L").resize(crop.size, Image.NEAREST)
        cutout = crop.convert("RGBA")
        cutout.putalpha(mask_pil)
        cutouts.append(cutout)
    return cutouts or [image_pil.convert("RGBA")]

# ---------- Risk bucketing for forecast ----------
def categorize_risk_forecast(top: List[dict], feats: dict, window_days: Optional[int] = None, **kwargs) -> tuple[List[dict], dict]:
    """
    Risk buckets for /predict-forecast (weather+region).
    Uses probability *and* whether the forecast supports the disease's weather pattern.
    """
    if not top:
        return top, {"overall": "low", "why": "no predictions"}

    top_sorted = sorted(top, key=lambda x: -x["final"])

    def g(k, default=np.nan):
        try:
            return float(feats.get(k, default))
        except Exception:
            return default

    RH_mean_7 = g("RH_mean_7")
    Rain_days_7 = g("Rain_days_7", 0)
    LWD_hours_7 = g("LWD_hours_7", 0)
    VPD_mean_7 = g("VPD_mean_7")
    T_max_7 = g("T_max_7")

    def weather_supports(disease: str) -> bool:
        d = disease.lower()
        if "anthracnose" in d or "leaf spot" in d:
            return (not np.isnan(RH_mean_7) and RH_mean_7 >= 80.0) and (Rain_days_7 >= 2 or LWD_hours_7 >= 10)
        if "scorch" in d:
            return (not np.isnan(VPD_mean_7) and VPD_mean_7 >= 1.5) and (not np.isnan(T_max_7) and T_max_7 >= 30.0) and (Rain_days_7 <= 1)
        return False

    def bucket(prob: float, supported: bool) -> str:
        if (prob >= 0.55 and supported) or (prob >= 0.65):
            return "high"
        if prob >= 0.35 or supported:
            return "medium"
        return "low"

    # Assign per-disease risk
    for t in top_sorted:
        supported = weather_supports(t["label"])
        t["risk"] = bucket(t["final"], supported)

    overall_bucket = top_sorted[0]["risk"]
    why = f"top1={top_sorted[0]['label']} p={top_sorted[0]['final']:.2f}, supported_weather={weather_supports(top_sorted[0]['label'])}"
    return top_sorted, {"overall": overall_bucket, "why": why}

# ========================
# API Schemas
# ========================
class PredictResponse(BaseModel):
    top: List[dict]
    overlay_png_base64: Optional[str] = None
    leaf_pct: Optional[float] = None
    spot_pct: Optional[float] = None
    weather_features: Optional[dict] = None
    risk_summary: Optional[dict] = None
    management: Optional[dict] = None
    daily: Optional[List[dict]] = None
    risks_by_disease: Optional[Dict[str, List[dict]]] = None
    # NEW: peak risk objects
    peak_risk_by_disease: Optional[Dict[str, dict]] = None  # label -> {"risk","date"}
    peak_risk_for_top: Optional[dict] = None                 # {"for","risk","date"}

# ========================
# Routes
# ========================
@app.get("/health")
def health():
    return {"status": "ok", "device": device, "universe": len(DISEASE_UNIVERSE)}

# ---------- NORMAL PREDICTION (IMG + OPTIONAL TEXT) ----------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    text: Optional[str] = Form(None),
    include_overlay: bool = Form(True),
):
    # Read image
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")

    # Overlay + coverage
    overlay_b64 = None
    coverage = {"leaf_pct": None, "spot_pct": None}
    if include_overlay:
        overlay_rgb, coverage = analyze_segmentation(img)
        overlay_b64 = png_base64_from_rgb(overlay_rgb)

    # Image → CLIP (universe)
    cutouts = segment_leaf_cutouts(img)
    p_img_u = image_probs_from_clip_universe(cutouts, tau_img=0.9)

    # Text → universe (optional)
    p_text_u = None
    if text is not None and text.strip():
        p_text_vec = predict_text_disease(text)
        p_text_u   = remap_text_probs_to_universe(p_text_vec)

    # Product-of-experts fusion (img + optional text)
    parts: List[Tuple[np.ndarray, float]] = []
    parts.append((p_img_u, W_IMG_DEFAULT))
    if p_text_u is not None:
        parts.append((p_text_u, W_TEXT_DEFAULT))

    p_final = combine_product_of_experts(parts)

    # Top-5
    idx = np.argsort(-p_final)[:5]
    top = [{"label": DISEASE_UNIVERSE[i], "final": float(p_final[i]), "rank": int(k)}
           for k, i in enumerate(idx, 1)]

    return {
        "top": top,
        "overlay_png_base64": overlay_b64,
        "leaf_pct": coverage["leaf_pct"],
        "spot_pct": coverage["spot_pct"],
        "weather_features": None,
        "risk_summary": None,
        "management": None,
        "daily": None,
        "risks_by_disease": None,
    }

# ---------- FORECAST-ONLY PREDICTION (MULTI-WINDOW 7/14) ----------
@app.post("/predict-forecast", response_model=PredictResponse)
async def predict_forecast_only(
    latitude: float = Form(...),
    longitude: float = Form(...),
    timezone: str = Form("UTC"),
    forecast_days: int = Form(16),
    state_code: Optional[str] = Form(None),
    # choose a single window to *display* in the legacy fields; defaults to 14
    window_days: Optional[int] = Form(None),        # e.g., 7 or 14
    # OR pass a CSV list to compute multiple windows at once (e.g., "7,14")
    windows: Optional[str] = Form(None),
    # Build tips for this label if present; otherwise top-1 of selected window
    focus_label: Optional[str] = Form(None),
):
    """
    Weather + region priors from forecast, now window-aware (7 and/or 14).

    Emits (legacy = for selected_window only, to keep compat):
      - top
      - daily
      - risks_by_disease
      - risk_summary
      - peak_risk_by_disease
      - peak_risk_for_top
      - peak_risk_for_focus
      - management (for focus or top, from selected_window)
      - focus

    New (multi-window) fields:
      - windows_used: [7,14]
      - selected_window: 7|14
      - by_window: {
          "7": { top, daily, risks_by_disease, risk_summary, peak_* },
          "14": { ... }
        }
      - weather_features_by_window: { "7": {...}, "14": {...} }
    """
    # --------------------------
    # 0) Inputs & window setup
    # --------------------------
    # Parse windows CSV or fall back to single window parameter or default [7,14]
    windows_used: list[int]
    if windows:
        try:
            windows_used = sorted({int(x.strip()) for x in windows.split(",") if x.strip()})
        except ValueError:
            windows_used = [7, 14]
    else:
        if window_days is not None:
            windows_used = [int(window_days)]
        else:
            windows_used = [7, 14]

    # Which window should populate the legacy fields?
    selected_window = (
        int(window_days) if window_days in windows_used
        else (14 if 14 in windows_used else windows_used[0])
    )

    # --------------------------
    # 1) Fetch forecast & tables
    # --------------------------
    raw = fetch_open_meteo_forecast(
        latitude, longitude, timezone_str=timezone, forecast_days=forecast_days
    )
    hourly_df, daily_df = build_forecast_tables(raw)

    # Precompute per-day basic features once
    per_day_feats = _compute_daily_basic_features(hourly_df, daily_df)  # [{date, ...}]

    # Convenience
    U = len(DISEASE_UNIVERSE)

    # --------------------------
    # 2) Compute blocks per window
    # --------------------------
    by_window: dict[int, dict] = {}
    weather_features_by_window: dict[int, dict] = {}

    for W in windows_used:
        # 2.1) Build window-specific forecast features (rolling/aggregates over next W days)
        feats = compute_forecast_features(hourly_df, daily_df, window_days=W)
        weather_features_by_window[W] = feats

        # 2.2) Weather prior universe (disease likelihoods from forecast + region prior)
        p_wx_u = build_weather_prior_universe_from_forecast(
            feats, kb_path=WEATHER_KB_PATH, state_code=state_code, floor=WX_FLOOR, window_days=W
        )

        # 2.3) Top-5 diseases by prior
        idx = np.argsort(-p_wx_u)[:5]
        top = [
            {"label": DISEASE_UNIVERSE[i], "final": float(p_wx_u[i]), "rank": int(k)}
            for k, i in enumerate(idx, 1)
        ]

        # quick lookup map from universe prob
        p_map = {DISEASE_UNIVERSE[i]: float(p_wx_u[i]) for i in range(U)}

        # 2.4) Weekly risk bucket per disease (window-aware)
        top, risk_summary = categorize_risk_forecast(top, feats, window_days=W)

        # 2.5) Per-day risks for next W days (limit to top-5)
        daily_out: list[dict] = []
        risks_by_disease: dict[str, list[dict]] = {}

        # NOTE: If you only want the next 7 calendar days regardless of W, slice here.
        horizon = min(W, len(per_day_feats))
        for dfeat in per_day_feats[:horizon]:
            date_str = dfeat["date"]
            risk_map: dict[str, str] = {}
            for t in top:
                lbl = t["label"]
                prob = p_map.get(lbl, 0.0)
                # window-aware day support check (pass W if your helper accepts it)
                supported = _weather_supports_day(lbl, dfeat, window_days=W)
                bucket = _bucket(prob, supported)  # "low" | "medium" | "high"
                risk_map[lbl] = bucket
                risks_by_disease.setdefault(lbl, []).append({"date": date_str, "risk": bucket})
            daily_out.append({"date": date_str, "risk_map": risk_map})

        # 2.6) Peak risk per disease (highest severity, earliest date on ties)
        peak_risk_by_disease: dict[str, dict] = {}
        for lbl, entries in risks_by_disease.items():
            if not entries:
                continue
            max_sev = max(_severity_rank(e.get("risk", "low")) for e in entries)
            at_peak = [e for e in entries if _severity_rank(e.get("risk", "low")) == max_sev]
            best_entry = min(at_peak, key=lambda e: e.get("date", "9999-12-31"))
            peak_risk_by_disease[lbl] = {"risk": best_entry["risk"], "date": best_entry["date"]}

        # 2.7) Decide focus label (prefer provided, else top-1 in this window)
        top_labels = {t["label"] for t in top}
        focus_clean = (focus_label or "").strip()
        target_label = None
        if focus_clean:
            for lbl in set(list(top_labels) + list(risks_by_disease.keys())):
                if lbl.lower() == focus_clean.lower():
                    target_label = lbl
                    break
        if not target_label:
            target_label = top[0]["label"] if top else None

        # 2.8) Peaks for top-1 and focus (window-local)
        peak_risk_for_top = None
        if top:
            best = top[0]["label"]
            peak_top = peak_risk_by_disease.get(best)
            if peak_top:
                peak_risk_for_top = {"for": best, "risk": peak_top["risk"], "date": peak_top["date"]}

        peak_risk_for_focus = None
        if target_label:
            peak_focus = peak_risk_by_disease.get(target_label)
            if peak_focus:
                peak_risk_for_focus = {"for": target_label, "risk": peak_focus["risk"], "date": peak_focus["date"]}

        # 2.9) Management tips for focus (use window-local peak or weekly fallback)
        management_block = None
        if target_label:
            weekly_risk = next((t.get("risk") for t in top if t["label"] == target_label), None)
            chosen_risk = (peak_risk_by_disease.get(target_label, {}) or {}).get("risk") or weekly_risk or "low"
            mgmt = get_management_for(target_label, chosen_risk)
            if mgmt:
                management_block = {
                    "for": target_label,
                    "risk": chosen_risk,
                    "tips": mgmt.get("tips", []),
                    "safety": mgmt.get("safety", {}),
                }

        # Collect window block
        by_window[W] = {
            "top": top,
            "risk_summary": risk_summary,
            "daily": daily_out,
            "risks_by_disease": risks_by_disease,
            "peak_risk_by_disease": peak_risk_by_disease,
            "peak_risk_for_top": peak_risk_for_top,
            "peak_risk_for_focus": peak_risk_for_focus,
            "focus": {"label": target_label} if target_label else None,
            "management": management_block,  # window-local
        }

    # ------------------------------------
    # 3) Fill legacy fields from selection
    # ------------------------------------
    sel = by_window[selected_window]
    return {
        # legacy fields = selected window (backward compatible)
        "top": sel["top"],
        "overlay_png_base64": None,
        "leaf_pct": None,
        "spot_pct": None,
        "weather_features": weather_features_by_window[selected_window],
        "risk_summary": sel["risk_summary"],
        "management": sel["management"],
        "daily": sel["daily"],
        "risks_by_disease": sel["risks_by_disease"],
        "peak_risk_by_disease": sel["peak_risk_by_disease"],
        "peak_risk_for_top": sel["peak_risk_for_top"],
        "peak_risk_for_focus": sel["peak_risk_for_focus"],
        "focus": sel["focus"],

        # new window-aware fields
        "windows_used": windows_used,
        "selected_window": selected_window,
        "by_window": {str(w): by_window[w] for w in windows_used},
        "weather_features_by_window": {str(w): weather_features_by_window[w] for w in windows_used},
    }