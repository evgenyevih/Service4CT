from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = [
    "path_to_study",
    "study_uid",
    "series_uid",
    "probability_of_pathology",
    "pathology",
    "processing_status",
    "time_of_processing",
]


def save_report(df: pd.DataFrame, output_path: str) -> None:
    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[REQUIRED_COLUMNS]
    df.to_excel(output_path, index=False)
