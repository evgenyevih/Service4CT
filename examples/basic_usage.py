#!/usr/bin/env python3
"""
Пример использования ИИ-сервиса для анализа КТ органов грудной клетки.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.infer_pipeline import run_inference
from src.models.cnn3d import CNN3DModel
from src.utils.config import load_yaml
import numpy as np


def example_basic_inference():
    """Пример базового инференса"""
    print("🔍 Пример базового инференса")
    print("=" * 50)
    
    # Загрузка конфигурации
    config = load_yaml("configs/config.yaml")
    print(f"✅ Конфигурация загружена: {config.get('project', {}).get('name', 'Неизвестно')}")
    
    # Создание модели
    model = CNN3DModel(
        checkpoint_path="weights/best_model.pth",
        device="cpu"  # Используем CPU для примера
    )
    print("✅ Модель инициализирована")
    
    # Создание тестовых данных (случайный 3D массив)
    test_volume = np.random.rand(64, 64, 64).astype(np.float32)
    print(f"✅ Тестовый объем создан: {test_volume.shape}")
    
    # Предсказание
    probability = model.predict_probability(test_volume)
    print(f"📊 Вероятность патологии: {probability:.3f}")
    print(f"🏥 Классификация: {'Патология' if probability > 0.5 else 'Норма'}")


def example_config_usage():
    """Пример работы с конфигурацией"""
    print("\n⚙️ Пример работы с конфигурацией")
    print("=" * 50)
    
    config = load_yaml("configs/config.yaml")
    
    # Настройки инференса
    inference_config = config.get('inference', {})
    print(f"Количество срезов: {inference_config.get('num_slices', 'Не указано')}")
    print(f"Окно HU: {inference_config.get('window', 'Не указано')}")
    
    # Настройки модели
    model_config = config.get('model', {})
    print(f"Архитектура: {model_config.get('backbone', 'Не указано')}")
    print(f"Размер изображения: {model_config.get('img_size', 'Не указано')}")


def example_directory_structure():
    """Пример проверки структуры директорий"""
    print("\n📁 Проверка структуры директорий")
    print("=" * 50)
    
    required_dirs = [
        "data/input_zips",
        "data/results", 
        "data/workdir",
        "data/training_data",
        "logs",
        "weights",
        "configs"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - создаем...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ {dir_path} - создана")


if __name__ == "__main__":
    print("🚀 Примеры использования Service4CT - ИИ-сервиса анализа КТ")
    print("=" * 60)
    
    try:
        # Проверка структуры
        example_directory_structure()
        
        # Работа с конфигурацией
        example_config_usage()
        
        # Базовый инференс
        example_basic_inference()
        
        print("\n✅ Все примеры выполнены успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
