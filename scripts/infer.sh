#!/bin/bash
# Скрипт для запуска инференса ИИ-сервиса анализа КТ

echo "🔍 Запуск Service4CT - инференс для анализа КТ органов грудной клетки"
echo "=================================================================="

# Проверка наличия входных данных
if [ ! -d "data/input_zips" ]; then
    echo "❌ Ошибка: Папка data/input_zips не найдена"
    echo "   Создайте папку и поместите в неё ZIP архивы с DICOM файлами"
    exit 1
fi

# Создание необходимых папок
mkdir -p data/results
mkdir -p logs

# Запуск инференса
echo "🚀 Запуск анализа..."
python -m src.main --mode infer --input_dir data/input_zips --output_path data/results/results.xlsx --config configs/config.yaml

# Проверка результата
if [ -f "data/results/results.xlsx" ]; then
    echo "✅ Анализ завершен успешно!"
    echo "📊 Результаты сохранены в: data/results/results.xlsx"
    echo "📝 Логи сохранены в: logs/app.log"
else
    echo "❌ Ошибка при выполнении анализа"
    exit 1
fi
