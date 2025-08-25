# OZON Recommendation System Project

## 📁 Структура проекта

```
ozon/
├── data/
│   ├── ml_ozon_recsys_test_for_participants/     # Исходные данные
│   └── processed/                                # Обработанные данные
├── src/
│   ├── features/                                 # Feature Engineering
│   ├── models/                                   # Модели ML
│   └── evaluation/                               # Оценка моделей
├── notebooks/                                    # Jupyter ноутбуки
├── config/                                       # Конфигурационные файлы
├── scripts/                                      # Скрипты для запуска
└── requirements.txt                              # Зависимости
```

## 🚀 Последовательность запуска

### 1. Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Linux/Mac)
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Загрузка и подготовка данных

```bash
# Проверка доступности данных
python scripts/check_data.py

# Просмотр структуры данных
python scripts/explore_data.py
```

### 3. Feature Engineering

```bash
# Основной пайплайн построения фичей
python src/features/build_features.py

# С опциями (рекомендуется для начала)
python src/features/build_features.py --sample_size 100000

# Только загрузка данных для анализа
python src/features/data_loader.py --analyze_only
```

### 4. Обучение моделей

```bash
# Базовое обучение (на всех данных)
python src/models/train_model.py

# С уменьшенным объемом данных
python src/models/train_model.py --sample_size 100000

# С балансировкой классов
python src/models/train_model.py --sample_size 100000 --balance_classes

# Только определенные модели
python src/models/train_model.py --models logistic lightgbm
```

### 5. Оценка результатов

```bash
# Анализ результатов обучения
python src/evaluation/analyze_results.py

# Сравнение моделей
python src/evaluation/compare_models.py

# Визуализация метрик
python src/evaluation/plot_metrics.py
```

## ⚙️ Параметры запуска

### Для build_features.py
```bash
--sample_size N          # Размер сэмпла (по умолчанию: все данные)
--output_path PATH       # Путь для сохранения результатов
--skip_validation        # Пропустить валидацию
```

### Для train_model.py
```bash
--sample_size N          # Размер сэмпла (100000, 500000)
--balance_classes        # Балансировка классов
--models TYPE            # Модели: all, logistic, forest, lightgbm
--test_size RATIO        # Размер тестовой выборки (0.2)
```

## 🔧 Утилиты

### Проверка окружения
```bash
python scripts/check_environment.py
```

### Поиск данных
```bash
python scripts/find_data_files.py
```

### Очистка данных
```bash
python scripts/clean_processed_data.py
```

## 📊 Мониторинг процесса

### Просмотр логов
```bash
# В реальном времени
tail -f logs/processing.log

# Статус обработки
python scripts/check_status.py
```

### Использование ресурсов
```bash
# Память и CPU
python scripts/monitor_resources.py

# Время выполнения
time python src/features/build_features.py --sample_size 100000
```

## 🐛 Поиск и решение проблем

### Если не хватает памяти
```bash
# Уменьшите размер данных
python src/features/build_features.py --sample_size 50000

# Или используйте более легкие модели
python src/models/train_model.py --models logistic --sample_size 50000
```

### Если ошибки импорта
```bash
# Переустановите зависимости
pip install -r requirements.txt --force-reinstall

# Или обновите конкретный пакет
pip install --upgrade pandas scikit-learn
```

### Если данные не находятся
```bash
# Поиск parquet файлов
python scripts/find_parquet_files.py

# Проверка путей в конфиге
python scripts/check_config_paths.py
```

## 🎯 Рекомендованный workflow

1. **Сначала маленький эксперимент:**
```bash
python src/features/build_features.py --sample_size 100000
python src/models/train_model.py --sample_size 100000 --balance_classes
```

2. **Если результаты хорошие - масштабируйте:**
```bash
python src/features/build_features.py --sample_size 500000
python src/models/train_model.py --sample_size 500000
```

3. **Полный запуск:**
```bash
python src/features/build_features.py
python src/models/train_model.py
```

## 📈 Интерпретация результатов

- **ROC-AUC > 0.7** - неплохо
- **ROC-AUC > 0.8** - хорошо  
- **ROC-AUC > 0.9** - отлично
- **ROC-AUC ~0.5** - как случайное угадывание

## 💡 Советы

1. Начинайте с маленьких sample sizes
2. Сначала добейтесь ROC-AUC > 0.7 на small data
3. Потом масштабируйте на все данные
4. Логируйте все эксперименты
5. Сохраняйте лучшие модели

## 🆘 Если все сломалось

```bash
# Полная очистка и переустановка
rm -rf venv/
rm -rf data/processed/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/features/build_features.py --sample_size 50000
```

**Лучшая команда для старта:**
```bash
python src/features/build_features.py --sample_size 100000
python src/models/train_model.py --sample_size 100000 --balance_classes
```