# Руководство по развертыванию - Service4CT

## Обзор

Данное руководство описывает различные способы развертывания Service4CT в продакшене, включая Docker, локальную установку и облачные платформы.

## 🐳 Docker развертывание

### Предварительные требования

- Docker 20.10+
- Docker Compose 2.0+
- Минимум 8GB RAM
- 10GB свободного места на диске

### Быстрое развертывание

```bash
# Клонирование репозитория
git clone https://github.com/evgenyevih/Service4CT.git
cd Service4CT

# Запуск инференса
docker compose up inference
```

### Сборка образа

```bash
# Сборка Docker образа
docker build -t service4ct:latest .

# Запуск контейнера
docker run -v $(pwd)/data:/app/data -v $(pwd)/weights:/app/weights service4ct:latest
```

### Docker Compose конфигурация

```yaml
# docker-compose.yml
version: '3.8'

services:
  inference:
    build: .
    container_name: ct-analysis-inference
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs
      - ./weights:/app/weights
    environment:
      - CUDA_VISIBLE_DEVICES=  # Отключение GPU
      - PYTHONPATH=/app
    command: bash -lc "python -m src.main --mode infer --input_dir data/input_zips --output_path data/results/results.xlsx"
    restart: "no"
```

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `CUDA_VISIBLE_DEVICES` | GPU устройства | `""` (CPU) |
| `PYTHONPATH` | Python путь | `/app` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |

## 🖥️ Локальное развертывание

### Системные требования

- **ОС**: Linux (Ubuntu 20.04+), macOS, Windows 10+
- **Python**: 3.11+
- **RAM**: 8GB+ (рекомендуется 16GB)
- **Диск**: 10GB+ свободного места
- **GPU**: NVIDIA с CUDA 12.9+ (опционально)

### Установка

#### 1. Клонирование репозитория

```bash
git clone https://github.com/evgenyevih/Service4CT.git
cd Service4CT
```

#### 2. Создание виртуального окружения

```bash
# С помощью conda (рекомендуется)
conda create -n service4ct python=3.11
conda activate service4ct

# Или с помощью venv
python -m venv service4ct
source service4ct/bin/activate  # Linux/macOS
# service4ct\Scripts\activate  # Windows
```

#### 3. Установка зависимостей

```bash
# Основные зависимости
pip install -r requirements.txt

# PyTorch с CUDA поддержкой
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

#### 4. Проверка установки

```bash
# Тест импорта
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pydicom; print('PyDICOM: OK')"

# Запуск тестов
python -m pytest tests/
```

### Конфигурация

#### 1. Настройка путей

```bash
# Создание необходимых директорий
mkdir -p data/{input_zips,results,training_data,workdir}
mkdir -p weights logs plots

# Установка прав доступа
chmod +x scripts/*.sh
```

#### 2. Настройка конфигурации

Отредактируйте `configs/config.yaml`:

```yaml
# Настройки для продакшена
inference:
  num_slices: 64
  window: [-1000, 400]
  normalize: true
  threshold: 0.5

training:
  batch_size: 16  # Уменьшить для ограниченной памяти
  lr: 1e-4
  epochs: 50
  early_stopping_patience: 10
```

### Запуск сервиса

#### Инференс

```bash
# Через скрипт
./scripts/infer.sh

# Напрямую
python -m src.main --mode infer --input_dir data/input_zips --output_path data/results/results.xlsx
```

#### Обучение

```bash
# Через скрипт
./scripts/train.sh

# Напрямую
python -m src.main --mode train
```

## ☁️ Облачное развертывание

### AWS EC2

#### 1. Создание инстанса

```bash
# Рекомендуемая конфигурация
Instance Type: g4dn.xlarge (GPU)
Storage: 50GB SSD
OS: Ubuntu 20.04 LTS
```

#### 2. Установка Docker

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Установка Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 3. Развертывание

```bash
# Клонирование и запуск
git clone https://github.com/evgenyevih/Service4CT.git
cd Service4CT
docker compose up -d inference
```

### Google Cloud Platform

#### 1. Создание VM

```bash
# Создание инстанса
gcloud compute instances create service4ct \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud
```

#### 2. Установка CUDA

```bash
# Установка NVIDIA драйверов
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2004-12-9-local_12.9.0-545.23.08-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-9-local_12.9.0-545.23.08-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Azure Container Instances

#### 1. Создание контейнера

```bash
# Создание группы ресурсов
az group create --name service4ct-rg --location eastus

# Создание контейнера
az container create \
  --resource-group service4ct-rg \
  --name service4ct \
  --image service4ct:latest \
  --cpu 2 \
  --memory 8 \
  --ports 8080
```

## 🔧 Конфигурация продакшена

### Оптимизация производительности

#### 1. GPU настройки

```yaml
# configs/config.yaml
inference:
  num_slices: 64
  window: [-1000, 400]
  normalize: true
  threshold: 0.4965  # Оптимальный порог

training:
  batch_size: 32     # Увеличить для GPU
  lr: 1e-4
  epochs: 100
  early_stopping_patience: 20
```

#### 2. Мониторинг ресурсов

```bash
# Мониторинг GPU
nvidia-smi

# Мониторинг памяти
free -h

# Мониторинг диска
df -h
```

### Логирование

#### 1. Настройка логов

```yaml
# configs/config.yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/app.log"
  max_size: "10MB"
  backup_count: 5
```

#### 2. Ротация логов

```bash
# Настройка logrotate
sudo nano /etc/logrotate.d/service4ct

# Содержимое файла
/path/to/service4ct/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 user user
}
```

### Безопасность

#### 1. Ограничение доступа

```bash
# Создание пользователя для сервиса
sudo useradd -r -s /bin/false service4ct

# Установка прав доступа
sudo chown -R service4ct:service4ct /opt/service4ct
sudo chmod 755 /opt/service4ct
```

#### 2. Firewall настройки

```bash
# UFW настройки
sudo ufw allow 22    # SSH
sudo ufw allow 80   # HTTP (если нужен веб-интерфейс)
sudo ufw enable
```

## 📊 Мониторинг и метрики

### Health Check

```python
# health_check.py
import os
import sys
sys.path.append('src')

def health_check():
    """Проверка состояния сервиса"""
    checks = {
        'model_exists': os.path.exists('weights/best_model.pth'),
        'config_exists': os.path.exists('configs/config.yaml'),
        'data_dir_exists': os.path.exists('data/input_zips'),
        'results_dir_writable': os.access('data/results', os.W_OK)
    }
    
    all_ok = all(checks.values())
    return all_ok, checks

if __name__ == "__main__":
    ok, checks = health_check()
    print(f"Health Check: {'PASS' if ok else 'FAIL'}")
    for check, status in checks.items():
        print(f"  {check}: {'✓' if status else '✗'}")
```

### Метрики производительности

```python
# metrics.py
import time
import psutil
import torch

def get_system_metrics():
    """Получение системных метрик"""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    }
```

## 🚀 Автоматизация развертывания

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Deploy Service4CT

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t service4ct:${{ github.sha }} .
      
      - name: Deploy to production
        run: |
          docker tag service4ct:${{ github.sha }} service4ct:latest
          docker-compose up -d inference
```

### Ansible Playbook

```yaml
# deploy.yml
- hosts: production
  become: yes
  tasks:
    - name: Install Docker
      apt:
        name: docker.io
        state: present
    
    - name: Install Docker Compose
      pip:
        name: docker-compose
        state: present
    
    - name: Clone repository
      git:
        repo: https://github.com/evgenyevih/Service4CT.git
        dest: /opt/service4ct
        version: main
    
    - name: Start service
      docker_compose:
        project_src: /opt/service4ct
        state: present
```

## 🔍 Устранение неполадок

### Частые проблемы

#### 1. Ошибки CUDA

```bash
# Проверка CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Решение: установка правильной версии PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

#### 2. Нехватка памяти

```bash
# Мониторинг памяти
free -h
ps aux --sort=-%mem | head

# Решение: уменьшение batch_size в конфигурации
```

#### 3. Ошибки DICOM

```python
# Проверка DICOM файлов
import pydicom
try:
    ds = pydicom.dcmread("file.dcm")
    print("DICOM файл корректен")
except Exception as e:
    print(f"Ошибка DICOM: {e}")
```

### Логи и отладка

```bash
# Просмотр логов
tail -f logs/app.log

# Отладка Docker
docker logs ct-analysis-inference

# Проверка статуса
docker ps -a
```

## 📈 Масштабирование

### Горизонтальное масштабирование

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  inference:
    build: .
    scale: 3  # Запуск 3 экземпляров
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - WORKER_ID=${WORKER_ID}
```

### Вертикальное масштабирование

```yaml
# Увеличение ресурсов
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

## 📋 Чек-лист развертывания

### Предварительная проверка

- [ ] Системные требования выполнены
- [ ] Docker установлен и работает
- [ ] Модель загружена в `weights/best_model.pth`
- [ ] Конфигурация настроена
- [ ] Тестовые данные подготовлены

### Развертывание

- [ ] Код склонирован
- [ ] Зависимости установлены
- [ ] Контейнер собран
- [ ] Сервис запущен
- [ ] Health check пройден

### Пост-развертывание

- [ ] Логирование настроено
- [ ] Мониторинг активен
- [ ] Резервное копирование настроено
- [ ] Документация обновлена

---

**Примечание**: Данное руководство предназначено для развертывания в различных средах. Выберите подходящий метод в зависимости от ваших требований и инфраструктуры.
