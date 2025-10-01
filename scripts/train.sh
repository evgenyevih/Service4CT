#!/bin/bash
# Скрипт для обучения модели ИИ-сервиса анализа КТ

echo "🚀 Запуск Service4CT - обучение модели для анализа КТ органов грудной клетки"
echo "========================================================================="

# Проверка наличия данных для обучения
if [ ! -f "data/training_data.csv" ]; then
    echo "❌ Ошибка: Файл data/training_data.csv не найден"
    echo "   Убедитесь, что данные для обучения находятся в правильной папке"
    exit 1
fi

if [ ! -d "data/training_data" ]; then
    echo "❌ Ошибка: Папка с данными для обучения не найдена"
    echo "   Убедитесь, что папка data/training_data существует"
    exit 1
fi

# Создание необходимых папок
mkdir -p logs
mkdir -p weights

# Запуск обучения
echo "🧠 Запуск обучения модели..."
python -m src.main --mode train --config configs/config.yaml

# Проверка результата
if [ -f "weights/best_model.pth" ]; then
    echo "✅ Обучение завершено успешно!"
    echo "🎯 Модель сохранена в: weights/best_model.pth"
    echo "📝 Логи сохранены в: logs/app.log"
else
    echo "❌ Ошибка при обучении модели"
    exit 1
fi
