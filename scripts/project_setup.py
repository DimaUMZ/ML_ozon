import json
from datetime import datetime

def setup_project():
    """Полная настройка проекта на основе EDA"""
    
    # Загрузка EDA
    with open('eda_results.json', 'r') as f:
        eda = json.load(f)
    
    # Создание директорий проекта
    import os
    directories = [
        'data/processed',
        'models',
        'notebooks',
        'src/features',
        'src/models',
        'src/evaluation',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Создание конфигурационных файлов
    config = {
        "project_name": "ecommerce-recommender",
        "version": "1.0",
        "created_date": datetime.now().isoformat(),
        "data_paths": {
            "raw_data": "data/raw/",
            "processed_data": "data/processed/",
            "features": "data/processed/features.parquet"
        },
        "validation": {
            "test_size_days": 30,
            "n_cv_splits": 3,
            "min_user_interactions": 3
        },
        "model_params": {
            "objective": "binary",
            "metric": "ndcg",
            "num_leaves": 31,
            "learning_rate": 0.05
        }
    }
    
    with open('config/project_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Проект успешно настроен!")
    print("Структура проекта:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py') or file.endswith('.json'):
                print(f"{subindent}{file}")

if __name__ == "__main__":
    setup_project()