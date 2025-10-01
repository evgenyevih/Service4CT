# Service4CT - ИИ-сервис для анализа компьютерных томографий органов грудной клетки

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Описание

ИИ-сервис для автоматического анализа компьютерных томографий (КТ) органов грудной клетки с целью классификации снимков на нормальные и патологические. Сервис использует глубокое обучение (3D CNN) для обработки DICOM файлов и выявления патологий.

## 🎯 Основные возможности

- **Автоматический анализ КТ**: Обработка DICOM файлов с автоматическим извлечением серий
- **Бинарная классификация**: Разделение на нормальные и патологические случаи
- **Поддержка Multi-frame DICOM**: Обработка сложных медицинских изображений
- **Простая интеграция**: Легкое подключение к внешним системам
- **Контейнеризация**: Полная поддержка Docker для развертывания
- **Детальная отчетность**: Excel отчеты с результатами анализа

## 🏗️ Архитектура

```
src/
├── io/                    # Модули для работы с DICOM файлами
├── models/                # Архитектуры нейронных сетей
├── pipelines/             # Пайплайны обучения и инференса
├── training/              # Скрипты обучения модели
├── utils/                 # Утилиты (логирование, отчеты, препроцессинг)
```

## 🚀 Быстрый старт

### Предварительные требования

- Python 3.11+
- CUDA (опционально, для GPU ускорения)
- Docker и Docker Compose (для контейнеризованного развертывания)

### Установка зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/evgenyevih/Service4CT.git
cd Service4CT

# Создание виртуального окружения
conda create -n service4ct python=3.11
conda activate service4ct

# Установка зависимостей
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### Запуск через Docker

```bash
# Запуск инференса
docker compose up inference
```

### Локальный запуск

```bash
# Активация окружения
conda activate service4ct

# Запуск инференса
./scripts/infer.sh

# Запуск обучения
./scripts/train.sh
```

## 📊 Использование

### Подготовка данных

1. Поместите ZIP архивы с DICOM файлами в папку `data/input_zips/`
2. Убедитесь, что архивы содержат корректные DICOM файлы

### Запуск анализа

```bash
# Через Docker
docker-compose up inference

# Локально
python -m src.main --mode infer --input_dir data/input_zips --output_path data/results/results.xlsx
```

### Результаты

Результаты анализа сохраняются в Excel файл со следующими колонками:
- `path_to_study` - путь к исследованию
- `study_uid` - уникальный идентификатор исследования
- `series_uid` - уникальный идентификатор серии
- `probability_of_pathology` - вероятность патологии (0-1)
- `pathology` - бинарная классификация (0=норма, 1=патология)
- `processing_status` - статус обработки
- `time_of_processing` - время обработки в секундах

## 🧠 Обучение модели

### Подготовка данных для обучения

1. Разместите CSV файл с метками в `data/training_data.csv`
2. Разместите папку с DICOM данными в `data/training_data/`

### Запуск обучения

```bash
# Локально
python -m src.main --mode train
```

### Особенности обучения

- **Улучшенная архитектура**: 3D CNN
- **Балансировка классов**: Weighted sampling и Focal Loss
- **Оптимизация порога**: Автоматический поиск оптимального порога классификации
- **Early stopping**: Предотвращение переобучения
- **Визуализация**: ROC кривые и метрики качества

## 🔧 Интеграция

Сервис может быть интегрирован в существующие системы через:
- Прямой вызов Python API
- Docker контейнеры
- Batch обработка файлов

## ⚙️ Конфигурация

Основные настройки находятся в `configs/config.yaml`

## 📁 Структура проекта

```
├── src/                           # Исходный код
│   ├── io/                       # DICOM I/O
│   ├── models/                   # Модели нейронных сетей
│   ├── pipelines/                # Пайплайны обработки
│   ├── training/                 # Скрипты обучения
│   └── utils/                    # Утилиты
├── configs/                      # Конфигурационные файлы
├── data/                         # Данные
│   ├── input_zips/              # Входные ZIP архивы
│   ├── results/                 # Результаты анализа
│   ├── training_data/           # DICOM данные для обучения
│   ├── training_data.csv        # Метки для обучения
│   └── workdir/                 # Рабочая директория
├── weights/                      # Веса обученных моделей
├── logs/                         # Логи приложения
├── scripts/                      # Скрипты запуска
├── docs/                         # Документация
├── tests/                        # Тесты
├── Dockerfile                    # Docker образ
├── docker-compose.yml            # Docker Compose
└── requirements.txt              # Python зависимости
```

## 🔧 Разработка

### Установка для разработки

```bash
# Клонирование и установка
git clone https://github.com/evgenyevih/Service4CT.git
cd Service4CT
conda create -n service4ct python=3.11
conda activate service4ct
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### Запуск тестов

```bash
# Запуск всех тестов
pytest tests/

# Запуск с покрытием
pytest --cov=src tests/
```

### Линтинг и форматирование

```bash
# Проверка стиля кода
flake8 src/
black src/
isort src/
```

## 📈 Производительность

- **Время обработки**: ~2-5 секунд на исследование
- **Точность**: AUC ~ 0.77 на тестовом наборе
- **Чувствительность**: ~ 0.919 для выявления патологий
- **Специфичность**: ~ 0.544 для нормальных случаев

## 🐛 Устранение неполадок

### Частые проблемы

1. **Ошибка CUDA**: Убедитесь, что CUDA установлена и совместима с PyTorch 2.8.0+cu129
2. **Недостаток памяти**: Уменьшите batch_size в конфигурации
3. **Ошибки DICOM**: Проверьте корректность DICOM файлов

### Логи

Логи сохраняются в `logs/app.log` с подробной информацией о процессе обработки.

## 📄 Лицензия

MIT License - см. файл [LICENSE](LICENSE) для подробностей.

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📚 Документация

Подробная документация доступна в папке `docs/`:

- **[docs/README.md](docs/README.md)** - Индекс документации
- **[docs/API.md](docs/API.md)** - API документация
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Руководство по развертыванию
- **[docs/TRAINING.md](docs/TRAINING.md)** - Руководство по обучению модели
- **[docs/EXAMPLES.md](docs/EXAMPLES.md)** - Примеры использования

## 📞 Поддержка

Для вопросов и поддержки:
- Создайте Issue в репозитории
- Обратитесь к документации в папке `docs/`

## 🔧 Технические требования

### Версии PyTorch
- **PyTorch**: 2.8.0+cu129 (с поддержкой CUDA 12.9)
- **Torchvision**: 0.23.0+cu129
- **Установка**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129`

### Системные требования
- **Python**: 3.11+
- **CUDA**: 12.9+ (опционально)
- **RAM**: 8GB+ (рекомендуется 16GB)
- **GPU**: NVIDIA с поддержкой CUDA (опционально)

## 🙏 Благодарности

- PyTorch за платформу глубокого обучения
- Pydicom за работу с DICOM файлами

---

**Внимание**: Данный сервис предназначен для исследовательских целей. Для клинического использования требуется дополнительная валидация и сертификация.