# API Документация - Service4CT

## Обзор

Service4CT предоставляет Python API для анализа компьютерных томографий органов грудной клетки. API поддерживает как инференс (анализ), так и обучение моделей.

## Основные модули

### 1. Главный модуль (`src.main`)

**Функция**: `main()`

Точка входа в приложение. Поддерживает два режима работы:

#### Параметры командной строки

```bash
python -m src.main [OPTIONS]
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--mode` | str | `infer` | Режим работы: `train` или `infer` |
| `--input_dir` | str | `data/input_zips` | Путь к папке с ZIP архивами |
| `--output_path` | str | `data/results/results.xlsx` | Путь для сохранения результатов |
| `--config` | str | `configs/config.yaml` | Путь к файлу конфигурации |

#### Примеры использования

```bash
# Инференс
python -m src.main --mode infer --input_dir data/input_zips --output_path results.xlsx

# Обучение
python -m src.main --mode train --config configs/config.yaml
```

### 2. Пайплайн инференса (`src.pipelines.infer_pipeline`)

**Функция**: `run_inference(input_dir, output_path, config_path)`

Обрабатывает ZIP архивы с DICOM файлами и возвращает результаты анализа.

#### Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `input_dir` | str | Путь к папке с ZIP архивами |
| `output_path` | str | Путь для сохранения Excel отчета |
| `config_path` | str | Путь к файлу конфигурации |

#### Возвращаемое значение

`None` - результаты сохраняются в Excel файл

#### Пример использования

```python
from src.pipelines.infer_pipeline import run_inference

# Запуск инференса
run_inference(
    input_dir="data/input_zips",
    output_path="data/results/results.xlsx",
    config_path="configs/config.yaml"
)
```

### 3. Пайплайн обучения (`src.pipelines.train_pipeline`)

**Функция**: `run_training(config_path)`

Обучает 3D CNN модель на предоставленных данных.

#### Параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `config_path` | str | Путь к файлу конфигурации |

#### Возвращаемое значение

`tuple` - (обученная модель, метрики качества)

#### Пример использования

```python
from src.pipelines.train_pipeline import run_training

# Запуск обучения
model, metrics = run_training("configs/config.yaml")
```

### 4. Модель CNN3D (`src.models.cnn3d`)

**Класс**: `CNN3DModel`

Обертка для 3D CNN модели с поддержкой загрузки весов и предсказаний.

#### Инициализация

```python
from src.models.cnn3d import CNN3DModel

model = CNN3DModel(
    checkpoint_path="weights/best_model.pth",
    device="cpu",  # или "cuda"
    depth_size=64,
    spatial_size=64
)
```

#### Методы

##### `predict(volume)`

Выполняет предсказание на 3D объеме.

**Параметры:**
- `volume` (numpy.ndarray): 3D массив изображения

**Возвращает:**
- `float`: Вероятность патологии (0-1)

##### `predict_batch(volumes)`

Обрабатывает батч 3D объемов.

**Параметры:**
- `volumes` (list): Список 3D массивов

**Возвращает:**
- `list`: Список вероятностей патологии

### 5. DICOM I/O (`src.io.dicom_io`)

#### Функции

##### `extract_zip(zip_path, extract_dir)`

Извлекает ZIP архив с DICOM файлами.

```python
from src.io.dicom_io import extract_zip

extract_zip("data/input_zips/study.zip", "data/workdir/")
```

##### `find_dicom_files(directory)`

Находит все DICOM файлы в директории.

```python
from src.io.dicom_io import find_dicom_files

dicom_files = find_dicom_files("data/workdir/")
```

##### `group_by_series(dicom_files)`

Группирует DICOM файлы по сериям.

```python
from src.io.dicom_io import group_by_series

series_groups = group_by_series(dicom_files)
```

### 6. Препроцессинг (`src.utils.preprocess`)

#### Функции

##### `series_to_normalized_slices(series_files, num_slices=64, window=(-1000, 400))`

Конвертирует серию DICOM файлов в нормализованные срезы.

```python
from src.utils.preprocess import series_to_normalized_slices

slices = series_to_normalized_slices(
    series_files,
    num_slices=64,
    window=(-1000, 400)
)
```

### 7. Отчеты (`src.utils.report`)

#### Функции

##### `save_inference_report(results, output_path)`

Сохраняет результаты инференса в Excel файл.

```python
from src.utils.report import save_inference_report

results = [
    {
        "path_to_study": "study1",
        "probability_of_pathology": 0.85,
        "pathology": 1,
        "processing_status": "success"
    }
]

save_inference_report(results, "results.xlsx")
```

## Конфигурация

### Файл конфигурации (`configs/config.yaml`)

```yaml
# Настройки инференса
inference:
  num_slices: 64                    # Количество срезов
  window: [-1000, 400]              # Окно HU
  normalize: true                    # Нормализация
  threshold: 0.5                    # Порог классификации

# Настройки обучения
training:
  batch_size: 32                    # Размер батча
  lr: 1e-4                         # Скорость обучения
  epochs: 100                       # Количество эпох
  weight_decay: 1e-4               # L2 регуляризация
  early_stopping_patience: 20       # Терпение для early stopping
  dropout_rate: 0.3                 # Dropout
  test_size: 0.2                    # Доля тестового набора
  val_size: 0.2                     # Доля валидационного набора
  random_state: 42                  # Случайное зерно
  n_bootstrap: 1000                 # Bootstrap выборки
  confidence_level: 0.95            # Уровень доверия
  focal_loss_alpha: 0.75           # Альфа для Focal Loss
  focal_loss_gamma: 2.0            # Гамма для Focal Loss
  class_weights: true               # Веса классов
  use_weighted_sampler: true        # Взвешенная выборка
  optimize_threshold: true          # Оптимизация порога
  series_dir: "data/training_data"  # Папка с данными
  data_file: "training_data.csv"    # Файл с метками

# Настройки модели
model:
  num_classes: 2                    # Количество классов
  checkpoint_path: "weights/best_model.pth"  # Путь к весам

# Логирование
logging:
  level: "INFO"                     # Уровень логирования
```

## Примеры использования

### Полный пример инференса

```python
import os
from src.pipelines.infer_pipeline import run_inference

# Настройка путей
input_dir = "data/input_zips"
output_path = "data/results/results.xlsx"
config_path = "configs/config.yaml"

# Проверка наличия входных данных
if not os.path.exists(input_dir):
    print(f"❌ Папка {input_dir} не найдена")
    exit(1)

if not os.listdir(input_dir):
    print(f"❌ Папка {input_dir} пуста")
    exit(1)

# Запуск инференса
print("🔍 Запуск анализа КТ...")
run_inference(input_dir, output_path, config_path)
print("✅ Анализ завершен!")
```

### Полный пример обучения

```python
from src.pipelines.train_pipeline import run_training

# Запуск обучения
print("🚀 Запуск обучения модели...")
model, metrics = run_training("configs/config.yaml")

print("✅ Обучение завершено!")
print(f"📊 Метрики: {metrics}")
```

### Работа с моделью напрямую

```python
import numpy as np
from src.models.cnn3d import CNN3DModel

# Инициализация модели
model = CNN3DModel(
    checkpoint_path="weights/best_model.pth",
    device="cpu"
)

# Создание тестового 3D объема
test_volume = np.random.rand(64, 64, 64).astype(np.float32)

# Предсказание
probability = model.predict(test_volume)
print(f"Вероятность патологии: {probability:.4f}")
```

## Обработка ошибок

### Типичные исключения

1. **FileNotFoundError**: Файл модели или конфигурации не найден
2. **ValueError**: Некорректные параметры конфигурации
3. **RuntimeError**: Ошибки CUDA или нехватка памяти
4. **DicomError**: Проблемы с чтением DICOM файлов

### Пример обработки ошибок

```python
try:
    from src.pipelines.infer_pipeline import run_inference
    run_inference("data/input_zips", "results.xlsx", "config.yaml")
except FileNotFoundError as e:
    print(f"❌ Файл не найден: {e}")
except ValueError as e:
    print(f"❌ Некорректные параметры: {e}")
except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")
```

## Производительность

### Рекомендации по оптимизации

1. **GPU ускорение**: Используйте CUDA для больших объемов данных
2. **Batch обработка**: Обрабатывайте несколько файлов одновременно
3. **Память**: Увеличьте RAM для больших DICOM файлов
4. **Параллелизация**: Используйте multiprocessing для множественных файлов

### Мониторинг производительности

```python
import time
from src.pipelines.infer_pipeline import run_inference

start_time = time.time()
run_inference("data/input_zips", "results.xlsx", "config.yaml")
end_time = time.time()

print(f"⏱️ Время обработки: {end_time - start_time:.2f} секунд")
```
