from __future__ import annotations
from src.training.train_model import train_model

def run_training(config_path: str) -> None:
    """Запуск обучения модели"""
    print(f"🚀 Запуск обучения модели...")
    print(f"📋 Конфигурация: {config_path}")
    
    try:
        model, metrics = train_model(config_path)
        print("✅ Обучение завершено успешно!")
        return model, metrics
    except Exception as e:
        print(f"❌ Ошибка во время обучения: {e}")
        raise
