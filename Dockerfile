# Dockerfile для ИИ-сервиса анализа КТ органов грудной клетки
FROM python:3.11-slim

# Настройка переменных окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libjpeg-dev \
    libopenjp2-7-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование и установка Python зависимостей
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p logs data/results data/workdir

# Установка прав доступа
RUN chmod +x scripts/*.sh

# Команда по умолчанию
CMD ["bash", "-lc", "python -m src.main"]
