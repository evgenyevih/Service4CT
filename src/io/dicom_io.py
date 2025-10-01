from __future__ import annotations
import os
import zipfile
from typing import Dict, List, Tuple

import pydicom
from pydicom.dataset import FileDataset


REQUIRED_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "InstanceNumber",
]


def extract_zip(zip_path: str, extract_to: str) -> str:
    name = os.path.splitext(os.path.basename(zip_path))[0]
    target_dir = os.path.join(extract_to, name)
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)
    return target_dir


def find_dicom_files(root: str) -> List[str]:
    dicoms: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                # Quick check: read header only
                dcm = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                if hasattr(dcm, "SOPClassUID"):
                    dicoms.append(fp)
            except Exception:
                continue
    return dicoms


def load_and_group_by_series(dicom_paths: List[str]) -> Dict[str, List[FileDataset]]:
    series_to_instances: Dict[str, List[FileDataset]] = {}
    for fp in dicom_paths:
        try:
            ds: FileDataset = pydicom.dcmread(fp, force=True)
        except Exception:
            continue
        if not all(hasattr(ds, tag) for tag in REQUIRED_TAGS):
            continue
        
        series_uid = str(ds.SeriesInstanceUID)
        
        # Проверяем, является ли это Multi-frame DICOM
        if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
            # Multi-frame DICOM - создаем только ОДИН экземпляр с метаданными
            # НЕ копируем pixel_array для экономии памяти!
            ds._is_multiframe = True
            ds._num_frames = ds.NumberOfFrames
            series_to_instances.setdefault(series_uid, []).append(ds)
        else:
            # Обычный single-frame DICOM
            series_to_instances.setdefault(series_uid, []).append(ds)
    
    # sort each series by InstanceNumber if present
    for k, lst in series_to_instances.items():
        lst.sort(key=lambda d: getattr(d, "InstanceNumber", 0))
    return series_to_instances


def get_uids(ds: FileDataset) -> Tuple[str, str]:
    return str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID)
