from __future__ import annotations
from typing import List, Tuple
import numpy as np
from pydicom.dataset import FileDataset


def to_hu(pixel_array: np.ndarray, ds: FileDataset) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixel_array.astype(np.float32) * slope + intercept
    return hu


def window_image(img_hu: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    wmin, wmax = float(window[0]), float(window[1])
    img = np.clip(img_hu, wmin, wmax)
    img = (img - wmin) / max(wmax - wmin, 1e-6)
    return img.astype(np.float32)


def select_slices(num_slices: int, total: int) -> List[int]:
    if total <= 0:
        return []
    if num_slices >= total:
        return list(range(total))
    idxs = np.linspace(0, total - 1, num=num_slices)
    return [int(round(i)) for i in idxs]


def series_to_normalized_slices(
    instances: List[FileDataset],
    num_slices: int = 64,
    window: Tuple[float, float] = (-1000.0, 400.0),
) -> np.ndarray:
    """
    Convert a DICOM series (sorted instances) to a stack of normalized slices in [0,1].
    Returns array of shape (S, H, W) where S<=num_slices.
    Updated for CNN3D model with Multi-frame DICOM support.
    """
    if not instances:
        return np.zeros((0, 1, 1), dtype=np.float32)

    slice_imgs: List[np.ndarray] = []
    
    for ds in instances:
        try:
            # Проверяем, является ли это Multi-frame DICOM
            if hasattr(ds, '_is_multiframe') and ds._is_multiframe:
                # Multi-frame DICOM - извлекаем все кадры и выбираем нужные
                pixel_array = ds.pixel_array  # Получаем весь массив (476, 512, 512)
                
                # Выбираем равномерно распределенные кадры
                num_frames = ds._num_frames
                if num_slices >= num_frames:
                    # Если нужно больше срезов чем есть кадров, берем все
                    selected_frames = list(range(num_frames))
                else:
                    # Выбираем равномерно распределенные кадры
                    step = num_frames / num_slices
                    selected_frames = [int(i * step) for i in range(num_slices)]
                
                # Обрабатываем выбранные кадры
                for frame_idx in selected_frames:
                    if frame_idx < num_frames:
                        frame_pixels = pixel_array[frame_idx]  # Извлекаем конкретный кадр
                        hu = to_hu(frame_pixels, ds)
                        win = window_image(hu, window)
                        slice_imgs.append(win)
            else:
                # Обычный single-frame DICOM
                pixel_array = ds.pixel_array
                hu = to_hu(pixel_array, ds)
                win = window_image(hu, window)
                slice_imgs.append(win)
            
        except Exception as e:
            print(f"Ошибка обработки среза: {e}")
            continue

    if not slice_imgs:
        return np.zeros((0, 1, 1), dtype=np.float32)

    total = len(slice_imgs)
    idxs = select_slices(num_slices, total)
    selected = [slice_imgs[i] for i in idxs]
    stack = np.stack(selected, axis=0) if selected else np.zeros((0, 1, 1), dtype=np.float32)
    return stack


