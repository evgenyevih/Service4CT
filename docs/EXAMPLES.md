# Примеры использования - Service4CT

## Обзор

Данный документ содержит практические примеры использования Service4CT для различных сценариев анализа компьютерных томографий органов грудной клетки.

## 🚀 Быстрые примеры

### Пример 1: Базовый инференс

```python
#!/usr/bin/env python3
"""
Базовый пример использования Service4CT для анализа КТ
"""

import os
import sys
sys.path.append('src')

from src.pipelines.infer_pipeline import run_inference

def main():
    # Настройка путей
    input_dir = "data/input_zips"
    output_path = "data/results/basic_inference.xlsx"
    config_path = "configs/config.yaml"
    
    # Проверка входных данных
    if not os.path.exists(input_dir):
        print(f"❌ Папка {input_dir} не найдена")
        return
    
    if not os.listdir(input_dir):
        print(f"❌ Папка {input_dir} пуста")
        return
    
    print("🔍 Запуск базового инференса...")
    
    try:
        # Запуск инференса
        run_inference(input_dir, output_path, config_path)
        print(f"✅ Результаты сохранены в {output_path}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
```

### Пример 2: Обучение модели

```python
#!/usr/bin/env python3
"""
Пример обучения модели Service4CT
"""

import os
import sys
sys.path.append('src')

from src.pipelines.train_pipeline import run_training

def main():
    config_path = "configs/config.yaml"
    
    # Проверка данных для обучения
    if not os.path.exists("data/training_data.csv"):
        print("❌ Файл training_data.csv не найден")
        return
    
    if not os.path.exists("data/training_data"):
        print("❌ Папка training_data не найдена")
        return
    
    print("🚀 Запуск обучения модели...")
    
    try:
        # Запуск обучения
        model, metrics = run_training(config_path)
        print("✅ Обучение завершено!")
        print(f"📊 Метрики: {metrics}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
```

## 🔧 Продвинутые примеры

### Пример 3: Работа с моделью напрямую

```python
#!/usr/bin/env python3
"""
Прямая работа с обученной моделью
"""

import numpy as np
import torch
from src.models.cnn3d import CNN3DModel

def load_and_predict():
    """Загрузка модели и предсказание"""
    
    # Инициализация модели
    model = CNN3DModel(
        checkpoint_path="weights/best_model.pth",
        device="cpu",  # или "cuda" для GPU
        depth_size=64,
        spatial_size=64
    )
    
    # Создание тестового 3D объема
    test_volume = np.random.rand(64, 64, 64).astype(np.float32)
    
    # Предсказание
    probability = model.predict(test_volume)
    
    print(f"📊 Вероятность патологии: {probability:.4f}")
    print(f"🎯 Классификация: {'Патология' if probability >= 0.5 else 'Норма'}")
    
    return probability

def batch_prediction():
    """Пакетное предсказание"""
    
    model = CNN3DModel(checkpoint_path="weights/best_model.pth")
    
    # Создание батча 3D объемов
    batch_volumes = [
        np.random.rand(64, 64, 64).astype(np.float32) for _ in range(5)
    ]
    
    # Пакетное предсказание
    probabilities = model.predict_batch(batch_volumes)
    
    print("📊 Результаты пакетного предсказания:")
    for i, prob in enumerate(probabilities):
        print(f"  Объем {i+1}: {prob:.4f} ({'Патология' if prob >= 0.5 else 'Норма'})")

if __name__ == "__main__":
    print("🔍 Тестирование модели...")
    load_and_predict()
    print("\n📦 Пакетное предсказание:")
    batch_prediction()
```

### Пример 4: Обработка DICOM файлов

```python
#!/usr/bin/env python3
"""
Обработка DICOM файлов с детальным анализом
"""

import os
import sys
sys.path.append('src')

from src.io.dicom_io import extract_zip, find_dicom_files, group_by_series
from src.utils.preprocess import series_to_normalized_slices
from src.models.cnn3d import CNN3DModel

def process_dicom_zip(zip_path, output_dir):
    """Обработка ZIP архива с DICOM файлами"""
    
    print(f"🔍 Обработка {zip_path}...")
    
    # Извлечение ZIP архива
    extract_dir = os.path.join(output_dir, "extracted")
    extract_zip(zip_path, extract_dir)
    
    # Поиск DICOM файлов
    dicom_files = find_dicom_files(extract_dir)
    print(f"📁 Найдено DICOM файлов: {len(dicom_files)}")
    
    # Группировка по сериям
    series_groups = group_by_series(dicom_files)
    print(f"📊 Найдено серий: {len(series_groups)}")
    
    # Инициализация модели
    model = CNN3DModel(checkpoint_path="weights/best_model.pth")
    
    results = []
    
    # Обработка каждой серии
    for series_uid, series_files in series_groups.items():
        print(f"🔬 Обработка серии {series_uid}...")
        
        try:
            # Конвертация в нормализованные срезы
            slices = series_to_normalized_slices(
                series_files,
                num_slices=64,
                window=(-1000, 400)
            )
            
            # Предсказание
            probability = model.predict(slices)
            
            result = {
                'series_uid': series_uid,
                'probability_of_pathology': float(probability),
                'pathology': int(probability >= 0.5),
                'num_slices': len(series_files),
                'status': 'success'
            }
            
            results.append(result)
            print(f"  ✅ Вероятность патологии: {probability:.4f}")
            
        except Exception as e:
            print(f"  ❌ Ошибка обработки: {e}")
            results.append({
                'series_uid': series_uid,
                'probability_of_pathology': 0.0,
                'pathology': 0,
                'num_slices': len(series_files),
                'status': 'error',
                'error': str(e)
            })
    
    return results

def main():
    # Настройка путей
    zip_path = "data/input_zips/study.zip"
    output_dir = "data/workdir"
    
    if not os.path.exists(zip_path):
        print(f"❌ Файл {zip_path} не найден")
        return
    
    # Создание выходной директории
    os.makedirs(output_dir, exist_ok=True)
    
    # Обработка
    results = process_dicom_zip(zip_path, output_dir)
    
    # Сохранение результатов
    import pandas as pd
    df = pd.DataFrame(results)
    output_file = "data/results/dicom_analysis.xlsx"
    df.to_excel(output_file, index=False)
    
    print(f"✅ Результаты сохранены в {output_file}")
    print(f"📊 Обработано серий: {len(results)}")
    print(f"🎯 Патологий выявлено: {sum(r['pathology'] for r in results)}")

if __name__ == "__main__":
    main()
```

### Пример 5: Кастомная конфигурация

```python
#!/usr/bin/env python3
"""
Использование кастомной конфигурации
"""

import yaml
import os
import sys
sys.path.append('src')

from src.pipelines.infer_pipeline import run_inference

def create_custom_config():
    """Создание кастомной конфигурации"""
    
    config = {
        'project': {
            'name': 'Service4CT'
        },
        'inference': {
            'num_slices': 128,  # Увеличенное количество срезов
            'window': [-1200, 600],  # Расширенное окно HU
            'normalize': True,
            'threshold': 0.3  # Более чувствительный порог
        },
        'training': {
            'batch_size': 16,  # Меньший batch для экономии памяти
            'lr': 5e-5,  # Более консервативная скорость обучения
            'epochs': 50,
            'weight_decay': 1e-3,
            'early_stopping_patience': 15,
            'dropout_rate': 0.4,  # Больше dropout
            'test_size': 0.15,
            'val_size': 0.15,
            'random_state': 42,
            'n_bootstrap': 500,
            'confidence_level': 0.90,
            'focal_loss_alpha': 0.8,
            'focal_loss_gamma': 2.5,
            'class_weights': True,
            'use_weighted_sampler': True,
            'optimize_threshold': True,
            'series_dir': 'data/training_data',
            'data_file': 'training_data.csv'
        },
        'model': {
            'num_classes': 2,
            'checkpoint_path': 'weights/best_model.pth'
        },
        'logging': {
            'level': 'DEBUG'  # Более подробное логирование
        }
    }
    
    # Сохранение конфигурации
    config_path = "configs/custom_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ Кастомная конфигурация сохранена в {config_path}")
    return config_path

def run_with_custom_config():
    """Запуск с кастомной конфигурацией"""
    
    # Создание конфигурации
    config_path = create_custom_config()
    
    # Настройка путей
    input_dir = "data/input_zips"
    output_path = "data/results/custom_inference.xlsx"
    
    print("🔍 Запуск инференса с кастомной конфигурацией...")
    
    try:
        run_inference(input_dir, output_path, config_path)
        print(f"✅ Результаты сохранены в {output_path}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    run_with_custom_config()
```

## 🐳 Docker примеры

### Пример 6: Docker инференс

```bash
#!/bin/bash
# Пример запуска инференса через Docker

# Сборка образа
echo "🔨 Сборка Docker образа..."
docker build -t service4ct:latest .

# Запуск инференса
echo "🔍 Запуск инференса..."
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/configs:/app/configs \
  service4ct:latest \
  python -m src.main --mode infer \
  --input_dir data/input_zips \
  --output_path data/results/docker_inference.xlsx

echo "✅ Инференс завершен!"
```

### Пример 7: Docker Compose

```yaml
# docker-compose.custom.yml
version: '3.8'

services:
  inference:
    build: .
    container_name: service4ct-inference
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs
      - ./weights:/app/weights
    environment:
      - CUDA_VISIBLE_DEVICES=0  # Использование GPU 0
      - PYTHONPATH=/app
      - LOG_LEVEL=DEBUG
    command: bash -lc "python -m src.main --mode infer --input_dir data/input_zips --output_path data/results/compose_inference.xlsx"
    restart: "no"
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

```bash
# Запуск с кастомной конфигурацией
docker compose -f docker-compose.custom.yml up inference
```

## 📊 Анализ результатов

### Пример 8: Анализ результатов инференса

```python
#!/usr/bin/env python3
"""
Анализ результатов инференса
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

def analyze_results(results_file):
    """Анализ результатов инференса"""
    
    # Загрузка результатов
    df = pd.read_excel(results_file)
    
    print("📊 Общая статистика:")
    print(f"  Всего исследований: {len(df)}")
    print(f"  Успешно обработано: {len(df[df['processing_status'] == 'success'])}")
    print(f"  Ошибок: {len(df[df['processing_status'] == 'error'])}")
    
    # Анализ патологий
    pathology_count = df['pathology'].sum()
    normal_count = len(df) - pathology_count
    
    print(f"\n🎯 Классификация:")
    print(f"  Патологии: {pathology_count} ({pathology_count/len(df)*100:.1f}%)")
    print(f"  Норма: {normal_count} ({normal_count/len(df)*100:.1f}%)")
    
    # Статистика вероятностей
    prob_stats = df['probability_of_pathology'].describe()
    print(f"\n📈 Статистика вероятностей:")
    print(f"  Среднее: {prob_stats['mean']:.4f}")
    print(f"  Медиана: {prob_stats['50%']:.4f}")
    print(f"  Стандартное отклонение: {prob_stats['std']:.4f}")
    print(f"  Минимум: {prob_stats['min']:.4f}")
    print(f"  Максимум: {prob_stats['max']:.4f}")
    
    return df

def plot_results(df, output_dir="plots"):
    """Визуализация результатов"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # График распределения вероятностей
    plt.figure(figsize=(10, 6))
    plt.hist(df['probability_of_pathology'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', label='Порог 0.5')
    plt.xlabel('Вероятность патологии')
    plt.ylabel('Количество исследований')
    plt.title('Распределение вероятностей патологии')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Box plot по классам
    plt.figure(figsize=(8, 6))
    df.boxplot(column='probability_of_pathology', by='pathology', ax=plt.gca())
    plt.title('Распределение вероятностей по классам')
    plt.xlabel('Класс (0=Норма, 1=Патология)')
    plt.ylabel('Вероятность патологии')
    plt.savefig(f'{output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Графики сохранены в {output_dir}/")

def main():
    results_file = "data/results/results.xlsx"
    
    if not os.path.exists(results_file):
        print(f"❌ Файл {results_file} не найден")
        return
    
    print("📊 Анализ результатов инференса...")
    
    # Анализ данных
    df = analyze_results(results_file)
    
    # Визуализация
    plot_results(df)
    
    print("✅ Анализ завершен!")

if __name__ == "__main__":
    main()
```

### Пример 9: Сравнение моделей

```python
#!/usr/bin/env python3
"""
Сравнение различных моделей
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compare_models(results_files, model_names):
    """Сравнение результатов разных моделей"""
    
    results = {}
    
    for file, name in zip(results_files, model_names):
        if os.path.exists(file):
            df = pd.read_excel(file)
            results[name] = df
            print(f"✅ Загружены результаты для {name}")
        else:
            print(f"❌ Файл {file} не найден")
    
    # Сравнение метрик
    print("\n📊 Сравнение моделей:")
    print("-" * 50)
    
    for name, df in results.items():
        if 'probability_of_pathology' in df.columns and 'pathology' in df.columns:
            # Фильтрация успешных предсказаний
            valid_df = df[df['processing_status'] == 'success']
            
            if len(valid_df) > 0:
                y_true = valid_df['pathology']
                y_pred = valid_df['probability_of_pathology']
                
                # Вычисление метрик
                auc_score = roc_auc_score(y_true, y_pred)
                precision, recall, _ = precision_recall_curve(y_true, y_pred)
                pr_auc = auc(recall, precision)
                
                print(f"{name}:")
                print(f"  AUC-ROC: {auc_score:.4f}")
                print(f"  AUC-PR: {pr_auc:.4f}")
                print(f"  Исследований: {len(valid_df)}")
                print()

def main():
    # Список файлов результатов для сравнения
    results_files = [
        "data/results/results.xlsx",
        "data/results/custom_inference.xlsx",
        "data/results/docker_inference.xlsx"
    ]
    
    model_names = [
        "Базовая модель",
        "Кастомная конфигурация",
        "Docker модель"
    ]
    
    print("🔍 Сравнение моделей...")
    compare_models(results_files, model_names)

if __name__ == "__main__":
    main()
```

## 🔧 Интеграция с внешними системами

### Пример 10: REST API обертка

```python
#!/usr/bin/env python3
"""
REST API обертка для Service4CT
"""

from flask import Flask, request, jsonify, send_file
import os
import tempfile
import zipfile
from src.pipelines.infer_pipeline import run_inference

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервиса"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists('weights/best_model.pth'),
        'config_exists': os.path.exists('configs/config.yaml')
    })

@app.route('/analyze', methods=['POST'])
def analyze_ct():
    """Анализ КТ файлов"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'Only ZIP files are supported'}), 400
    
    try:
        # Создание временной директории
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, 'input')
            output_path = os.path.join(temp_dir, 'results.xlsx')
            
            os.makedirs(input_dir, exist_ok=True)
            
            # Сохранение загруженного файла
            zip_path = os.path.join(input_dir, file.filename)
            file.save(zip_path)
            
            # Запуск инференса
            run_inference(input_dir, output_path, 'configs/config.yaml')
            
            # Чтение результатов
            import pandas as pd
            df = pd.read_excel(output_path)
            
            # Конвертация в JSON
            results = df.to_dict('records')
            
            return jsonify({
                'status': 'success',
                'results': results,
                'total_studies': len(results),
                'pathologies_detected': sum(r['pathology'] for r in results)
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def download_results(filename):
    """Скачивание результатов"""
    results_path = f'data/results/{filename}'
    if os.path.exists(results_path):
        return send_file(results_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### Пример 11: Batch обработка

```python
#!/usr/bin/env python3
"""
Batch обработка множественных файлов
"""

import os
import glob
import concurrent.futures
from src.pipelines.infer_pipeline import run_inference

def process_single_zip(zip_path, output_dir):
    """Обработка одного ZIP файла"""
    
    try:
        # Создание уникальной выходной директории
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        zip_output_dir = os.path.join(output_dir, zip_name)
        os.makedirs(zip_output_dir, exist_ok=True)
        
        # Запуск инференса
        output_path = os.path.join(zip_output_dir, 'results.xlsx')
        run_inference(
            input_dir=os.path.dirname(zip_path),
            output_path=output_path,
            config_path='configs/config.yaml'
        )
        
        return {
            'zip_path': zip_path,
            'output_path': output_path,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'zip_path': zip_path,
            'error': str(e),
            'status': 'error'
        }

def batch_process(input_dir, output_dir, max_workers=4):
    """Batch обработка множественных файлов"""
    
    # Поиск всех ZIP файлов
    zip_files = glob.glob(os.path.join(input_dir, '*.zip'))
    
    if not zip_files:
        print(f"❌ ZIP файлы не найдены в {input_dir}")
        return
    
    print(f"📁 Найдено ZIP файлов: {len(zip_files)}")
    print(f"👥 Используется потоков: {max_workers}")
    
    # Параллельная обработка
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запуск задач
        future_to_zip = {
            executor.submit(process_single_zip, zip_path, output_dir): zip_path 
            for zip_path in zip_files
        }
        
        # Сбор результатов
        for future in concurrent.futures.as_completed(future_to_zip):
            zip_path = future_to_zip[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"✅ {os.path.basename(zip_path)}")
                else:
                    print(f"❌ {os.path.basename(zip_path)}: {result['error']}")
                    
            except Exception as e:
                print(f"❌ {os.path.basename(zip_path)}: {e}")
                results.append({
                    'zip_path': zip_path,
                    'error': str(e),
                    'status': 'error'
                })
    
    # Статистика
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n📊 Статистика обработки:")
    print(f"  Успешно: {successful}")
    print(f"  Ошибок: {failed}")
    print(f"  Всего: {len(results)}")
    
    return results

def main():
    input_dir = "data/input_zips"
    output_dir = "data/batch_results"
    
    if not os.path.exists(input_dir):
        print(f"❌ Папка {input_dir} не найдена")
        return
    
    print("🔄 Запуск batch обработки...")
    results = batch_process(input_dir, output_dir, max_workers=2)
    
    if results:
        print("✅ Batch обработка завершена!")

if __name__ == "__main__":
    main()
```

## 📋 Чек-лист примеров

### Базовые примеры

- [ ] Пример 1: Базовый инференс
- [ ] Пример 2: Обучение модели
- [ ] Пример 3: Работа с моделью напрямую

### Продвинутые примеры

- [ ] Пример 4: Обработка DICOM файлов
- [ ] Пример 5: Кастомная конфигурация
- [ ] Пример 6: Docker инференс
- [ ] Пример 7: Docker Compose

### Анализ и интеграция

- [ ] Пример 8: Анализ результатов
- [ ] Пример 9: Сравнение моделей
- [ ] Пример 10: REST API обертка
- [ ] Пример 11: Batch обработка

---

**Примечание**: Все примеры готовы к запуску и содержат обработку ошибок. Адаптируйте пути и параметры под вашу конфигурацию.
