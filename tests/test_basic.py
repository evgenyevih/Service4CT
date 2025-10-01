#!/usr/bin/env python3
"""
Базовые тесты для ИИ-сервиса анализа КТ.
"""

import unittest
import os
import sys
import numpy as np

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn3d import CNN3DModel
from src.utils.config import load_yaml
from src.io.dicom_io import extract_zip, find_dicom_files
from src.utils.preprocess import series_to_normalized_slices


class TestBasicFunctionality(unittest.TestCase):
    """Тесты базовой функциональности"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.test_volume = np.random.rand(64, 64, 64).astype(np.float32)
        
    def test_model_initialization(self):
        """Тест инициализации модели"""
        model = CNN3DModel(device="cpu")
        self.assertIsNotNone(model)
        self.assertEqual(model.device.type, "cpu")
        
    def test_model_prediction(self):
        """Тест предсказания модели"""
        model = CNN3DModel(device="cpu")
        probability = model.predict_probability(self.test_volume)
        
        self.assertIsInstance(probability, float)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        
    def test_config_loading(self):
        """Тест загрузки конфигурации"""
        config = load_yaml("configs/config.yaml")
        self.assertIsInstance(config, dict)
        self.assertIn('project', config)
        self.assertIn('inference', config)
        self.assertIn('model', config)
        
    def test_preprocessing(self):
        """Тест препроцессинга"""
        # Создаем тестовые данные
        test_slices = np.random.rand(10, 64, 64).astype(np.float32)
        
        # Тестируем функцию выбора срезов
        from src.utils.preprocess import select_slices
        selected = select_slices(5, 10)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(0 <= i < 10 for i in selected))
        
    def test_empty_volume(self):
        """Тест обработки пустого объема"""
        model = CNN3DModel(device="cpu")
        empty_volume = np.array([])
        probability = model.predict_probability(empty_volume)
        
        # Должна возвращаться вероятность 0.5 для пустого входа
        self.assertEqual(probability, 0.5)


class TestDataStructure(unittest.TestCase):
    """Тесты структуры данных"""
    
    def test_required_directories(self):
        """Тест наличия необходимых директорий"""
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
            with self.subTest(directory=dir_path):
                self.assertTrue(os.path.exists(dir_path), 
                              f"Директория {dir_path} не найдена")
                
    def test_config_file(self):
        """Тест наличия файла конфигурации"""
        config_path = "configs/config.yaml"
        self.assertTrue(os.path.exists(config_path), 
                       f"Файл конфигурации {config_path} не найден")
        
    def test_requirements_file(self):
        """Тест наличия файла зависимостей"""
        requirements_path = "requirements.txt"
        self.assertTrue(os.path.exists(requirements_path),
                       f"Файл {requirements_path} не найден")


if __name__ == '__main__':
    # Создаем необходимые директории для тестов
    os.makedirs("data/input_zips", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/workdir", exist_ok=True)
    os.makedirs("data/training_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # Запуск тестов
    unittest.main(verbosity=2)
