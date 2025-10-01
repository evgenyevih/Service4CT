#!/usr/bin/env python3
"""
Service4CT - скрипт обучения модели для анализа КТ органов грудной клетки.
Включает оптимизацию порога, улучшенную балансировку классов и Focal Loss.
"""

import os
import warnings
from collections import Counter
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
from scipy import stats
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# Константы для обучения
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = ['Normal', 'Pathology']
TARGET_SIZE = (64, 64, 64)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Конфигурация обучения (значения по умолчанию)
CONFIG = {
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 20,
    'dropout_rate': 0.3,
    'device': DEVICE,
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42,
    'n_bootstrap': 1000,
    'confidence_level': 0.95,
    'focal_loss_alpha': 0.75,
    'focal_loss_gamma': 2.0,
    'class_weights': True,
    'use_weighted_sampler': True,
    'optimize_threshold': True,
    'series_dir': 'data/training_data',
    'data_file': 'training_data.csv'
}



class DICOMProcessor:
    """Процессор DICOM серий в нормализованные 3D объемы"""
    
    def __init__(self, target_size: Tuple[int, int, int] = TARGET_SIZE):
        self.target_size = target_size
        self.scaler = StandardScaler()
    
    def load_dicom_series(self, series_path: str) -> np.ndarray:
        """Загрузка и обработка DICOM серии в 3D объем"""
        try:
            # Получение всех DICOM файлов
            dicom_files = []
            for root, _, files in os.walk(series_path):
                for file in files:
                    dicom_files.append(os.path.join(root, file))
            
            if not dicom_files:
                raise ValueError(f"No DICOM files found in {series_path}")
            
            # Загрузка DICOM с улучшенной обработкой ошибок
            dicoms_with_position = []
            for filepath in dicom_files:
                try:
                    ds = pydicom.dcmread(filepath, force=True)
                    if hasattr(ds, 'PixelData'):
                        # Извлечение Z-позиции с надежной обработкой ошибок
                        z_pos = None
                        if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                            try:
                                z_pos = float(ds.ImagePositionPatient[2])
                            except (ValueError, TypeError):
                                pass
                        dicoms_with_position.append((ds, filepath, z_pos))
                except Exception:
                    continue
            
            if not dicoms_with_position:
                raise ValueError(f"No valid DICOM files with pixel data in {series_path}")
            
            # Сортировка по Z-позиции или InstanceNumber
            valid_slices = [(ds, filepath, z_pos) for ds, filepath, z_pos in dicoms_with_position if z_pos is not None]
            
            if not valid_slices:
                # Fallback к InstanceNumber
                dicoms_with_position.sort(key=lambda x: getattr(x[0], 'InstanceNumber', 0))
                dicoms = [(ds, filepath) for ds, filepath, _ in dicoms_with_position]
            else:
                # Сортировка по физической Z-позиции
                valid_slices.sort(key=lambda x: x[2])
                dicoms = [(ds, filepath) for ds, filepath, _ in valid_slices]
            
            # Извлечение объема с проверкой формы
            volume_slices = []
            for ds, _ in dicoms:
                try:
                    # Получение массива пикселей с обработкой ошибок
                    pixel_array = ds.pixel_array.astype(np.float32)
                    
                    # Проверка согласованности формы
                    if len(volume_slices) > 0 and pixel_array.shape != volume_slices[0].shape:
                        continue
                    
                    # Применение rescale если доступно
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        slope = float(ds.RescaleSlope)
                        intercept = float(ds.RescaleIntercept)
                        pixel_array = pixel_array * slope + intercept

                    volume_slices.append(pixel_array)
                except Exception:
                    continue
            
            if not volume_slices:
                raise ValueError("No valid slices extracted")
            
            # Проверка согласованности всех срезов перед объединением
            if len(set(slice.shape for slice in volume_slices)) > 1:
                shape_counts = Counter(slice.shape for slice in volume_slices)
                target_shape = shape_counts.most_common(1)[0][0]
                volume_slices = [slice for slice in volume_slices if slice.shape == target_shape]
            
            if not volume_slices:
                raise ValueError("No consistent slices found")
            
            # Объединение в 3D объем
            volume = np.stack(volume_slices, axis=0)  # Shape: (depth, height, width)
            
            # Нормализация и изменение размера
            volume = self.preprocess_volume(volume)
            
            return volume
            
        except Exception as e:
            print(f"Error processing series {series_path}: {e}")
            return np.zeros(self.target_size, dtype=np.float32)
    
    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Предобработка 3D объема: нормализация, обрезка, изменение размера"""
        if volume.size == 0:
            return np.zeros(self.target_size, dtype=np.float32)
        
        # Обрезка экстремальных значений (устойчиво к выбросам)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Нормализация к [0, 1]
        volume_min, volume_max = volume.min(), volume.max()
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        # Изменение размера до целевого размера
        if volume.shape != self.target_size:
            zoom_factors = [
                self.target_size[i] / volume.shape[i] for i in range(3)
            ]
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        return volume.astype(np.float32)



class CNN3D(nn.Module):
    """3D CNN с архитектурой для медицинских изображений"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(CNN3D, self).__init__()
        
        # Более сложная архитектура с residual connections
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Более глубокий классификатор
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate * 0.6)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов модели"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 1, depth, height, width]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Классификатор
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return torch.sigmoid(x)



class CTDataset(Dataset):
    """Датасет для загрузки данных обучения КТ"""
    
    def __init__(self, data_df: pd.DataFrame, series_dir: str, processor: DICOMProcessor):
        self.data_df = data_df
        self.series_dir = series_dir
        self.processor = processor
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        series_id = row[ID_COL]
        
        # Загрузка объема
        series_path = os.path.join(self.series_dir, series_id)
        volume = self.processor.load_dicom_series(series_path)
        
        # Получение меток
        labels = row[LABEL_COLS].values.astype(np.float32)
        
        # Конвертация в тензор и добавление канального размера
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)
        labels_tensor = torch.from_numpy(labels)
        
        return volume_tensor, labels_tensor



class FocalLoss(nn.Module):
    """Focal Loss для борьбы с дисбалансом классов"""
    
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Применение sigmoid к входам
        inputs = torch.sigmoid(inputs)
        
        # Вычисление BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Вычисление p_t
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Вычисление alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Вычисление focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Применение focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



def find_optimal_threshold(y_true, y_pred_proba):
    """Находит оптимальный порог на основе F1-score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Вычисление F1-score для каждого порога
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Поиск порога с максимальным F1-score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold, f1_scores[optimal_idx]

def calculate_metrics_with_threshold(y_true, y_pred_proba, threshold=0.5):
    """Вычисляет метрики с заданным порогом"""
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'threshold': threshold,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def save_training_results(all_metrics, optimal_threshold):
    """Сохраняет результаты обучения в текстовом формате"""
    from datetime import datetime
    
    # Создание отчета
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛИ Service4CT")
    report_lines.append("=" * 80)
    report_lines.append(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Оптимальный порог (на валидации): {optimal_threshold:.4f}")
    report_lines.append("")
    
    # Результаты по наборам данных
    for dataset_name, data in all_metrics.items():
        metrics = data['metrics']
        roc_auc = data['roc_auc']
        
        report_lines.append(f"{'=' * 60}")
        report_lines.append(f"РЕЗУЛЬТАТЫ НА {dataset_name.upper()} НАБОРЕ")
        report_lines.append(f"{'=' * 60}")
        report_lines.append(f"Порог: {metrics['threshold']:.4f}")
        report_lines.append(f"AUC-ROC: {roc_auc:.4f}")
        report_lines.append(f"Чувствительность: {metrics['sensitivity']:.4f}")
        report_lines.append(f"Специфичность: {metrics['specificity']:.4f}")
        report_lines.append(f"Точность: {metrics['precision']:.4f}")
        report_lines.append(f"F1-мера: {metrics['f1_score']:.4f}")
        report_lines.append(f"True Positives: {metrics['tp']}")
        report_lines.append(f"True Negatives: {metrics['tn']}")
        report_lines.append(f"False Positives: {metrics['fp']}")
        report_lines.append(f"False Negatives: {metrics['fn']}")
        report_lines.append("")
    
    # Сводная таблица
    report_lines.append("=" * 80)
    report_lines.append("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    report_lines.append("=" * 80)
    report_lines.append(f"{'Набор':<12} {'AUC-ROC':<8} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<10} {'F1-Score':<10}")
    report_lines.append("-" * 80)
    
    for dataset_name, data in all_metrics.items():
        metrics = data['metrics']
        roc_auc = data['roc_auc']
        report_lines.append(f"{dataset_name:<12} {roc_auc:<8.4f} {metrics['sensitivity']:<12.4f} {metrics['specificity']:<12.4f} {metrics['precision']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    report_lines.append("=" * 80)
    
    # Сохранение в файл
    report_content = "\n".join(report_lines)
    report_path = "plots/training_results.txt"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📄 Результаты обучения сохранены в: {report_path}")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить результаты: {e}")



def create_weighted_sampler(train_df):
    """Создает взвешенную выборку для балансировки классов"""
    class_counts = train_df['Pathology'].value_counts()
    class_weights = 1.0 / class_counts.values
    sample_weights = [class_weights[train_df.iloc[i]['Pathology']] for i in range(len(train_df))]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True
    )

def calculate_class_weights(train_df):
    """Рассчитывает веса классов для балансировки"""
    class_counts = train_df['Pathology'].value_counts()
    total_samples = len(train_df)
    
    # Обратно пропорциональные веса
    weights = []
    for class_id in [0, 1]:  # Normal, Pathology
        weight = total_samples / (len(class_counts) * class_counts[class_id])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)



class EarlyStopping:
    """Ранняя остановка для предотвращения переобучения"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()



def split_data(train_df, test_size=0.2, val_size=0.2, random_state=42):
    """Разделяет данные на train/val/test с сохранением баланса классов"""
    print("Разделение данных на train/val/test...")
    
    # Сначала отделяем тестовый набор
    train_val_df, test_df = train_test_split(
        train_df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=train_df['Pathology']
    )
    
    # Затем разделяем оставшиеся данные на train/val
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size_adjusted, 
        random_state=random_state,
        stratify=train_val_df['Pathology']
    )
    
    print(f"Разделение завершено:")
    print(f"• Обучающий набор: {len(train_df)} образцов")
    print(f"• Валидационный набор: {len(val_df)} образцов") 
    print(f"• Тестовый набор: {len(test_df)} образцов")
    
    # Проверка баланса классов
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        normal_count = (df['Pathology'] == 0).sum()
        pathology_count = (df['Pathology'] == 1).sum()
        print(f"• {name} - Normal: {normal_count}, Pathology: {pathology_count}")
    
    return train_df, val_df, test_df


def calculate_auc_during_training(model, dataloader, device):
    """Расчет AUC во время обучения"""
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # Берем только Normal класс (индекс 0)
            all_y_true.extend(labels[:, 0].cpu().numpy())
            all_y_pred.extend(outputs[:, 0].cpu().numpy())
    
    if len(np.unique(all_y_true)) > 1:
        return roc_auc_score(all_y_true, all_y_pred)
    else:
        return 0.5



def train_model(config_path: str = "configs/config.yaml"):
    """Основная функция обучения модели"""
    
    # Загрузка конфигурации из YAML файла
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # Обновление CONFIG значениями из YAML
        if 'training' in yaml_config:
            training_config = yaml_config['training']
            
            # Функция для безопасного преобразования типов
            def safe_convert(value, default, convert_func):
                if value is None:
                    return default
                try:
                    return convert_func(value)
                except (ValueError, TypeError):
                    return default
            
            CONFIG.update({
                'batch_size': safe_convert(training_config.get('batch_size'), CONFIG['batch_size'], int),
                'epochs': safe_convert(training_config.get('epochs'), CONFIG['epochs'], int),
                'learning_rate': safe_convert(training_config.get('lr'), CONFIG['learning_rate'], float),
                'weight_decay': safe_convert(training_config.get('weight_decay'), CONFIG['weight_decay'], float),
                'early_stopping_patience': safe_convert(training_config.get('early_stopping_patience'), CONFIG['early_stopping_patience'], int),
                'dropout_rate': safe_convert(training_config.get('dropout_rate'), CONFIG['dropout_rate'], float),
                'test_size': safe_convert(training_config.get('test_size'), CONFIG['test_size'], float),
                'val_size': safe_convert(training_config.get('val_size'), CONFIG['val_size'], float),
                'random_state': safe_convert(training_config.get('random_state'), CONFIG['random_state'], int),
                'n_bootstrap': safe_convert(training_config.get('n_bootstrap'), CONFIG['n_bootstrap'], int),
                'confidence_level': safe_convert(training_config.get('confidence_level'), CONFIG['confidence_level'], float),
                'focal_loss_alpha': safe_convert(training_config.get('focal_loss_alpha'), CONFIG['focal_loss_alpha'], float),
                'focal_loss_gamma': safe_convert(training_config.get('focal_loss_gamma'), CONFIG['focal_loss_gamma'], float),
                'class_weights': training_config.get('class_weights', CONFIG['class_weights']),
                'use_weighted_sampler': training_config.get('use_weighted_sampler', CONFIG['use_weighted_sampler']),
                'optimize_threshold': training_config.get('optimize_threshold', CONFIG['optimize_threshold']),
                'series_dir': training_config.get('series_dir', CONFIG['series_dir']),
                'data_file': training_config.get('data_file', CONFIG['data_file'])
            })
        
        print(f"📋 Конфигурация загружена из: {config_path}")
        print(f"• Эпохи: {CONFIG['epochs']}")
        print(f"• Batch size: {CONFIG['batch_size']}")
        print(f"• Learning rate: {CONFIG['learning_rate']}")
        
    except Exception as e:
        print(f"⚠️ Не удалось загрузить конфигурацию из {config_path}: {e}")
        print("🔄 Используется конфигурация по умолчанию")
    
    # Создание необходимых папок
    os.makedirs('plots', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 Начинаем обучение модели с улучшенной чувствительностью")
    print("="*80)
    
    # Загрузка данных
    print("📊 Загрузка данных...")
    train_df = pd.read_csv("data/training_data.csv")
    print(f"Загружено {len(train_df)} образцов")
    
    # Создание бинарной метки Pathology
    print("🔧 Создание бинарной метки Pathology...")
    train_df["Pathology"] = 0
    train_df.loc[train_df["Normal"] == 1, "Pathology"] = 0
    train_df.loc[(train_df["COVID"] == 1) | (train_df["Cancer"] == 1), "Pathology"] = 1
    
    # Статистика по бинарным классам
    normal_count = (train_df['Pathology'] == 0).sum()
    pathology_count = (train_df['Pathology'] == 1).sum()
    print(f"Бинарная классификация:")
    print(f"  Normal: {normal_count} записей")
    print(f"  Pathology: {pathology_count} записей")
    print(f"  Дисбаланс: {pathology_count/normal_count:.2f}:1")
    
    # Разделение данных
    train_df, val_df, test_df = split_data(
        train_df, 
        test_size=CONFIG['test_size'], 
        val_size=CONFIG['val_size'], 
        random_state=CONFIG['random_state']
    )
    
    # Расчет весов классов
    class_weights = calculate_class_weights(train_df)
    print(f"Веса классов: Normal={class_weights[0]:.3f}, Pathology={class_weights[1]:.3f}")
    
    # Инициализация компонентов
    print("🏗️ Инициализация модели...")
    processor = DICOMProcessor(TARGET_SIZE)
    model = CNN3D(num_classes=len(LABEL_COLS), dropout_rate=CONFIG['dropout_rate'])
    model.to(CONFIG['device'])
    print("✅ Используется модель CNN3D")
    
    # Создание датасетов
    train_dataset = CTDataset(train_df, "data/training_data", processor)
    val_dataset = CTDataset(val_df, "data/training_data", processor)
    test_dataset = CTDataset(test_df, "data/training_data", processor)
    
    # Создание взвешенной выборки для обучения
    if CONFIG.get('use_weighted_sampler', True):
        weighted_sampler = create_weighted_sampler(train_df)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=weighted_sampler, num_workers=4)
        print("✅ Используется взвешенная выборка для балансировки классов")
    else:
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # Функция потерь
    if CONFIG.get('class_weights', True):
        criterion = nn.BCELoss(weight=class_weights.to(CONFIG['device']))
        print("✅ Используется BCE Loss с весами классов")
    else:
        criterion = FocalLoss(
            alpha=CONFIG.get('focal_loss_alpha', 0.75),
            gamma=CONFIG.get('focal_loss_gamma', 2.0)
        )
        print("✅ Используется Focal Loss для борьбы с дисбалансом")
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    
    # История обучения
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': []
    }
    
    best_val_loss = float('inf')
    best_model_path = 'weights/best_model.pth'
    
    print(f"\n🎯 Начинаем обучение:")
    print(f"• Эпохи: {CONFIG['epochs']}")
    print(f"• Batch size: {CONFIG['batch_size']}")
    print(f"• Learning rate: {CONFIG['learning_rate']}")
    print(f"• Early stopping: {CONFIG['early_stopping_patience']}")
    
    # Обучение
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 40)
        
        # Обучение
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        train_loop = tqdm(train_loader, desc="Обучение", leave=False)
        for images, labels in train_loop:
            images = images.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            train_loop.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / train_samples
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="Валидация", leave=False)
            for images, labels in val_loop:
                images = images.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                val_loop.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / val_samples
        
        # Расчет AUC
        train_auc = calculate_auc_during_training(model, train_loader, CONFIG['device'])
        val_auc = calculate_auc_during_training(model, val_loader, CONFIG['device'])
        
        # Обновление истории
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        
        # Обновление scheduler
        scheduler.step(avg_val_loss)
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Сохранена лучшая модель (val_loss: {best_val_loss:.4f})")
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f"🛑 Early stopping на эпохе {epoch + 1}")
            break
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
    
    print(f"\n✅ Обучение завершено!")
    print(f"• Лучшая валидационная потеря: {best_val_loss:.4f}")
    print(f"• Обучено эпох: {len(history['train_loss'])}")
    

    
    print(f"\n📊 Оценка с оптимизацией порога...")
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load(best_model_path))
    
    # Сначала получаем предсказания для всех наборов
    datasets = {
        'train': (train_loader, "Обучающий"),
        'val': (val_loader, "Валидационный"), 
        'test': (test_loader, "Тестовый")
    }
    
    all_predictions = {}
    
    # Получение предсказаний для всех наборов
    for dataset_name, (loader, display_name) in datasets.items():
        print(f"\n🔍 Получение предсказаний для {display_name.lower()} набора:")
        
        model.eval()
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Предсказания {display_name.lower()}"):
                images = images.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])
                
                outputs = model(images)
                all_y_true.extend(labels.cpu().numpy())
                all_y_pred.extend(outputs.cpu().numpy())
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        # Для бинарной классификации берем только Normal класс (индекс 0)
        y_true_binary = all_y_true[:, 0]  # Normal labels
        y_pred_binary = all_y_pred[:, 0]  # Normal predictions
        
        all_predictions[dataset_name] = {
            'y_true': y_true_binary,
            'y_pred': y_pred_binary,
            'display_name': display_name
        }
    
    # Оптимизация порога ТОЛЬКО на валидационной выборке
    if CONFIG.get('optimize_threshold', True):
        val_data = all_predictions['val']
        optimal_threshold, optimal_f1 = find_optimal_threshold(val_data['y_true'], val_data['y_pred'])
        print(f"\n🎯 Оптимальный порог (на валидации): {optimal_threshold:.4f} (F1: {optimal_f1:.4f})")
        
        # Сохранение оптимального порога в конфигурацию
        import yaml
        config_path = "configs/config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            config['inference']['threshold'] = float(optimal_threshold)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            print(f"💾 Оптимальный порог сохранен в конфигурацию: {optimal_threshold:.4f}")
        except Exception as e:
            print(f"⚠️ Не удалось сохранить порог в конфигурацию: {e}")
    else:
        optimal_threshold = 0.5
        print(f"\n🎯 Используется стандартный порог: {optimal_threshold:.4f}")
    
    # Применение оптимального порога ко всем наборам
    all_metrics = {}
    
    for dataset_name, data in all_predictions.items():
        y_true_binary = data['y_true']
        y_pred_binary = data['y_pred']
        display_name = data['display_name']
        
        print(f"\n🔍 Оценка на {display_name.lower()} наборе:")
        
        # Расчет метрик с оптимальным порогом
        metrics = calculate_metrics_with_threshold(y_true_binary, y_pred_binary, optimal_threshold)
        
        # Вывод результатов
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ НА {display_name.upper()} НАБОРЕ")
        print(f"{'='*60}")
        print(f"• Порог: {optimal_threshold:.4f}")
        print(f"• Чувствительность: {metrics['sensitivity']:.3f}")
        print(f"• Специфичность: {metrics['specificity']:.3f}")
        print(f"• Точность: {metrics['precision']:.3f}")
        print(f"• F1-мера: {metrics['f1_score']:.3f}")
        print(f"• True Positives: {metrics['tp']}")
        print(f"• True Negatives: {metrics['tn']}")
        print(f"• False Positives: {metrics['fp']}")
        print(f"• False Negatives: {metrics['fn']}")
        print(f"{'='*60}")
        
        # ROC кривая
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.axvline(x=1-optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal threshold = {optimal_threshold:.4f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve - {display_name} Set')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'plots/roc_curve_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Сохранение метрик
        all_metrics[dataset_name] = {
            'metrics': metrics,
            'y_true': y_true_binary,
            'y_pred': y_pred_binary,
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc
        }
    
    # Финальные выводы
    test_metrics = all_metrics['test']['metrics']
    print(f"\n{'='*80}")
    print(f"ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА ТЕСТОВОМ НАБОРЕ")
    print(f"{'='*80}")
    print(f"• Оптимальный порог: {all_metrics['test']['optimal_threshold']:.4f}")
    print(f"• AUC-ROC: {all_metrics['test']['roc_auc']:.3f}")
    print(f"• Чувствительность: {test_metrics['sensitivity']:.3f}")
    print(f"• Специфичность: {test_metrics['specificity']:.3f}")
    print(f"• Точность: {test_metrics['precision']:.3f}")
    print(f"• F1-мера: {test_metrics['f1_score']:.3f}")
    print(f"{'='*80}")
    
    # Сохранение результатов в текстовом формате
    save_training_results(all_metrics, optimal_threshold)
    
    return model, all_metrics



if __name__ == "__main__":
    print("🧠 Service4CT - Обучение модели для детекции патологий КТ органов грудной клетки")
    print("📊 С улучшенной чувствительностью и оптимизацией порога")
    print("="*80)
    
    try:
        model, metrics = train_model()
        print("\n🎉 Обучение и оценка завершены успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()