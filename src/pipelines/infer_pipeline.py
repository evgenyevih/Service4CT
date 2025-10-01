from __future__ import annotations
import os
import time
import traceback
from typing import List

import pandas as pd
import torch

from src.io.dicom_io import extract_zip, find_dicom_files, load_and_group_by_series, get_uids
from src.utils.report import save_report
from loguru import logger
from src.utils.preprocess import series_to_normalized_slices
from src.models.cnn3d import CNN3DModel
from src.utils.config import load_yaml


def process_zip(zip_path: str, workdir: str, model: CNN3DModel | None = None, threshold: float = 0.5) -> List[dict]:
    start = time.time()
    out_rows: List[dict] = []
    try:
        extracted = extract_zip(zip_path, workdir)
        dcm_files = find_dicom_files(extracted)
        series_map = load_and_group_by_series(dcm_files)
        model = model or CNN3DModel()
        for series_uid, instances in series_map.items():
            if not instances:
                continue
            study_uid, _ = get_uids(instances[0])
            # Preprocess series -> normalized slice stack
            slices = series_to_normalized_slices(instances, num_slices=64, window=(-1000, 400))
            probability_of_pathology = model.predict_probability(slices)
            pathology = int(probability_of_pathology >= threshold)
            row = {
                "path_to_study": extracted,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "probability_of_pathology": probability_of_pathology,
                "pathology": pathology,
                "processing_status": "Success",
                "time_of_processing": round(time.time() - start, 3),
            }
            out_rows.append(row)
    except Exception as e:
        logger.exception(f"Failed to process zip {zip_path}")
        row = {
            "path_to_study": zip_path,
            "study_uid": "",
            "series_uid": "",
            "probability_of_pathology": 0.0,
            "pathology": 0,
            "processing_status": f"Failure: {str(e)}",
            "time_of_processing": round(time.time() - start, 3),
        }
        out_rows.append(row)
    return out_rows


def run_inference(input_dir: str, output_path: str, config_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    workdir = "data/workdir"
    os.makedirs(workdir, exist_ok=True)
    cfg = {}
    try:
        cfg = load_yaml(config_path)
    except Exception:
        logger.warning(f"Failed to load config {config_path}, using defaults")
    model_cfg = cfg.get("cnn3d", {}) if isinstance(cfg, dict) else {}

    # Prepare model from config
    model = CNN3DModel(
        checkpoint_path=model_cfg.get("checkpoint", "best_model.pth"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        depth_size=model_cfg.get("depth_size", 64),
        spatial_size=model_cfg.get("spatial_size", 64),
    )

    # Get threshold from config
    inference_cfg = cfg.get("inference", {}) if isinstance(cfg, dict) else {}
    threshold = inference_cfg.get("threshold", 0.5)
    logger.info(f"Using threshold: {threshold}")

    rows: List[dict] = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".zip"):
            continue
        zip_path = os.path.join(input_dir, fname)
        try:
            rows.extend(process_zip(zip_path, workdir, model=model, threshold=threshold))
        except Exception:
            logger.exception(f"Unhandled failure on file {zip_path}")

    df = pd.DataFrame(rows)
    save_report(df, output_path)
    logger.info(f"Saved report to {output_path} with {len(df)} rows")
