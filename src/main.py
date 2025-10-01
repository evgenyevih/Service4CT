"""
Service4CT - главный модуль ИИ-сервиса для анализа компьютерных томографий органов грудной клетки.
Поддерживает обучение модели и инференс на DICOM файлах.
"""

import argparse
from src.pipelines.infer_pipeline import run_inference
from src.pipelines.train_pipeline import run_training
from src.utils.logging_setup import setup_logging


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Service4CT - ИИ-сервис для анализа КТ органов грудной клетки")
    parser.add_argument("--mode", choices=["train", "infer"], default="infer", 
                       help="Режим работы: train (обучение) или infer (инференс)")
    parser.add_argument("--input_dir", type=str, default="data/input_zips",
                       help="Путь к папке с входными ZIP архивами")
    parser.add_argument("--output_path", type=str, default="data/results/results.xlsx",
                       help="Путь для сохранения результатов")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Путь к файлу конфигурации")
    return parser.parse_args()


def main():
    """Главная функция приложения"""
    args = parse_args()
    setup_logging("logs")
    
    if args.mode == "train":
        print("🚀 Запуск обучения модели...")
        run_training(args.config)
    else:
        print("🔍 Запуск инференса...")
        run_inference(args.input_dir, args.output_path, args.config)


if __name__ == "__main__":
    main()
