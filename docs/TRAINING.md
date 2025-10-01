# Руководство по обучению модели - Service4CT

## Обзор

Данное руководство описывает процесс обучения 3D CNN модели для анализа компьютерных томографий органов грудной клетки. Модель обучается на размеченных данных для бинарной классификации (норма/патология).

## 📊 Подготовка данных

### Структура данных

```
data/
├── training_data/           # DICOM файлы для обучения
│   ├── study_001/           # Исследование 1
│   │   ├── series_001/      # Серия 1
│   │   └── series_002/      # Серия 2
│   └── study_002/           # Исследование 2
└── training_data.csv        # Файл с метками
```

### Формат меток (training_data.csv)

```csv
study_uid,series_uid,pathology,age,sex
1.2.3.4.5.6.7.8.9.10,1.2.3.4.5.6.7.8.9.11,1,65,M
1.2.3.4.5.6.7.8.9.12,1.2.3.4.5.6.7.8.9.13,0,45,F
```

**Обязательные колонки:**
- `study_uid`: Уникальный идентификатор исследования
- `series_uid`: Уникальный идентификатор серии
- `pathology`: Метка класса (0=норма, 1=патология)

**Дополнительные колонки:**
- `age`: Возраст пациента
- `sex`: Пол пациента (M/F)

### Требования к DICOM файлам

1. **Формат**: Стандартные DICOM файлы
2. **Размер**: Рекомендуется 512x512 пикселей
3. **Толщина среза**: 1-5 мм
4. **Количество срезов**: Минимум 32, рекомендуется 64+
5. **Модальность**: CT (Computed Tomography)

## 🏗️ Архитектура модели

### CNN3D - 3D Convolutional Neural Network

```python
class CNN3D(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(CNN3D, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        
        # Pooling
        self.pool = nn.MaxPool3d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
```

### Особенности архитектуры

- **3D свертки**: Обработка объемных данных
- **Batch Normalization**: Стабилизация обучения
- **Dropout**: Предотвращение переобучения
- **Residual connections**: Улучшение градиентного потока

## ⚙️ Конфигурация обучения

### Основные параметры

```yaml
# configs/config.yaml
training:
  batch_size: 32                    # Размер батча
  lr: 1e-4                         # Скорость обучения
  epochs: 100                       # Количество эпох
  weight_decay: 1e-4               # L2 регуляризация
  early_stopping_patience: 20       # Терпение для early stopping
  dropout_rate: 0.3                 # Коэффициент dropout
  
  # Разделение данных
  test_size: 0.2                    # Доля тестового набора
  val_size: 0.2                     # Доля валидационного набора
  random_state: 42                  # Случайное зерно
  
  # Метрики и валидация
  n_bootstrap: 1000                 # Bootstrap выборки
  confidence_level: 0.95            # Уровень доверия
  
  # Потеря и балансировка
  focal_loss_alpha: 0.75           # Альфа для Focal Loss
  focal_loss_gamma: 2.0           # Гамма для Focal Loss
  class_weights: true               # Веса классов
  use_weighted_sampler: true        # Взвешенная выборка
  
  # Оптимизация порога
  optimize_threshold: true          # Оптимизация порога классификации
  
  # Пути к данным
  series_dir: "data/training_data"  # Папка с DICOM данными
  data_file: "training_data.csv"    # Файл с метками
```

### Продвинутые настройки

#### 1. Аугментация данных

```python
# Пример аугментации в train_model.py
transforms = [
    RandomRotation3D(degrees=10),
    RandomFlip3D(axis=2),  # Горизонтальное отражение
    RandomNoise3D(noise_factor=0.1),
    RandomBrightness3D(brightness_factor=0.2)
]
```

#### 2. Learning Rate Scheduler

```python
# Планировщик скорости обучения
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5,
    verbose=True
)
```

## 🚀 Запуск обучения

### Локальное обучение

```bash
# Активация окружения
conda activate service4ct

# Запуск обучения
python -m src.main --mode train

# Или через скрипт
./scripts/train.sh
```

### Docker обучение

```bash
# Запуск в контейнере
docker compose run --rm training

# Или с GPU
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/weights:/app/weights service4ct:latest python -m src.main --mode train
```

### Обучение с кастомными параметрами

```python
# Прямой вызов функции обучения
from src.training.train_model import train_model

# Обучение с кастомной конфигурацией
model, metrics = train_model("configs/custom_config.yaml")
```

## 📈 Мониторинг обучения

### Метрики в реальном времени

```python
# Логирование метрик
import wandb

# Инициализация Weights & Biases
wandb.init(project="service4ct", name="experiment_1")

# Логирование метрик
wandb.log({
    "epoch": epoch,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_auc": val_auc,
    "learning_rate": optimizer.param_groups[0]['lr']
})
```

### Визуализация процесса

```python
# ROC кривые
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('plots/roc_curve.png')
```

## 🎯 Оптимизация гиперпараметров

### Grid Search

```python
# Поиск оптимальных параметров
param_grid = {
    'lr': [1e-5, 1e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'dropout_rate': [0.2, 0.3, 0.4],
    'weight_decay': [1e-5, 1e-4, 1e-3]
}

best_params = grid_search(param_grid, X_train, y_train)
```

### Bayesian Optimization

```python
# Оптимизация с помощью Optuna
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    model = CNN3D(dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Обучение и валидация
    val_auc = train_and_validate(model, optimizer, batch_size)
    return val_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## 📊 Оценка качества модели

### Метрики классификации

```python
# Основные метрики
from sklearn.metrics import classification_report, confusion_matrix

# Отчет по классификации
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pathology']))

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
```

### Bootstrap валидация

```python
# Доверительные интервалы для метрик
def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000):
    metrics = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        auc = roc_auc_score(y_true_boot, y_pred_boot)
        metrics.append(auc)
    
    return np.percentile(metrics, [2.5, 97.5])  # 95% доверительный интервал
```

### Кросс-валидация

```python
# K-fold кросс-валидация
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Обучение модели на fold
    model = CNN3D()
    # ... обучение ...
    
    # Оценка на валидации
    val_auc = evaluate_model(model, X_val_fold, y_val_fold)
    cv_scores.append(val_auc)
    
    print(f"Fold {fold + 1}: AUC = {val_auc:.4f}")

print(f"CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

## 🔧 Устранение проблем

### Частые проблемы

#### 1. Переобучение (Overfitting)

**Симптомы:**
- Высокая точность на тренировочных данных
- Низкая точность на валидационных данных
- Большая разница между train и val loss

**Решения:**
```python
# Увеличение dropout
dropout_rate = 0.5

# Ранняя остановка
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

# Регуляризация
weight_decay = 1e-3

# Аугментация данных
transforms = [RandomRotation3D(), RandomFlip3D(), RandomNoise3D()]
```

#### 2. Недообучение (Underfitting)

**Симптомы:**
- Низкая точность на всех данных
- Модель не улучшается

**Решения:**
```python
# Увеличение сложности модели
hidden_size = 1024

# Уменьшение dropout
dropout_rate = 0.1

# Увеличение количества эпох
epochs = 200

# Увеличение learning rate
lr = 1e-3
```

#### 3. Дисбаланс классов

**Симптомы:**
- Модель предсказывает только один класс
- Низкая чувствительность или специфичность

**Решения:**
```python
# Веса классов
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

# Focal Loss
focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

# Взвешенная выборка
weighted_sampler = WeightedRandomSampler(weights, len(weights))
```

### Оптимизация производительности

#### 1. Ускорение обучения

```python
# Mixed Precision Training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 2. Экономия памяти

```python
# Gradient Accumulation
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📋 Чек-лист обучения

### Предварительная подготовка

- [ ] Данные подготовлены и размечены
- [ ] DICOM файлы проверены на корректность
- [ ] CSV файл с метками создан
- [ ] Конфигурация настроена
- [ ] Окружение активировано

### Процесс обучения

- [ ] Модель инициализирована
- [ ] Данные загружены и разделены
- [ ] Обучение запущено
- [ ] Метрики отслеживаются
- [ ] Визуализация работает

### Пост-обработка

- [ ] Лучшая модель сохранена
- [ ] Результаты проанализированы
- [ ] Порог оптимизирован
- [ ] Отчеты созданы
- [ ] Модель протестирована

## 📊 Анализ результатов

### Сохранение результатов

```python
# Сохранение детальных результатов
results = {
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'test_metrics': test_metrics,
    'optimal_threshold': optimal_threshold,
    'training_history': history
}

# Сохранение в файл
import json
with open('plots/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Интерпретация метрик

```python
# Анализ важности признаков
def analyze_feature_importance(model, dataloader):
    model.eval()
    gradients = []
    
    for batch in dataloader:
        batch.requires_grad_()
        output = model(batch)
        loss = output.sum()
        loss.backward()
        
        gradients.append(batch.grad.numpy())
    
    return np.mean(gradients, axis=0)
```

---

**Примечание**: Данное руководство покрывает основные аспекты обучения модели. Для достижения лучших результатов рекомендуется экспериментировать с различными архитектурами, гиперпараметрами и техниками аугментации данных.
