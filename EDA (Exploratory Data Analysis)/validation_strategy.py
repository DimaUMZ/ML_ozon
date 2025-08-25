# validation_strategy.py
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json

class ValidationStrategy:
    """
    Стратегия валидации для рекомендательной системы
    """
    
    def __init__(self, eda_results):
        self.eda = eda_results
        self.strategy_date = datetime.now().strftime("%Y-%m-%d")
        
    def create_validation_plan(self):
        """Создание полного плана валидации"""
        return {
            "validation_approach": "Временная валидация (Time-based split)",
            "rationale": "Рекомендательные системы подвержены временному дрейфу, временной сплит наиболее реалистичен",
            "train_test_split": self._get_time_split_strategy(),
            "cross_validation": self._get_cv_strategy(),
            "evaluation_metrics": self._get_evaluation_metrics(),
            "baseline_models": self._get_baseline_models(),
            "ab_testing_plan": self._get_ab_testing_plan()
        }
    
    def _get_time_split_strategy(self):
        """Стратегия разделения данных по времени"""
        return {
            "train_period": {
                "start": "2025-01-01",
                "end": "2025-05-31",
                "duration": "5 месяцев"
            },
            "test_period": {
                "start": "2025-06-01", 
                "end": "2025-07-01",
                "duration": "1 месяц"
            },
            "validation_ratio": "80/20 (временной сплит)",
            "min_user_interactions": 3,  # Минимальное число взаимодействий для включения в валидацию
            "warm_start_requirements": {
                "min_user_interactions_train": 2,
                "min_item_interactions_train": 5
            }
        }
    
    def _get_cv_strategy(self):
        """Стратегия кросс-валидации"""
        return {
            "method": "Time Series Cross-Validation",
            "n_splits": 3,
            "test_size": "1 месяц",
            "gap": "0",  # Без gap между train и test
            "validation_focus": "Проверка на временной устойчивости"
        }
    
    def _get_evaluation_metrics(self):
        """Метрики для оценки качества"""
        return {
            "ranking_metrics": [
                {"metric": "NDCG@10", "weight": 0.4, "purpose": "Качество ранжирования топ-10"},
                {"metric": "Precision@5", "weight": 0.3, "purpose": "Точность топ-5 рекомендаций"},
                {"metric": "Recall@20", "weight": 0.3, "purpose": "Полнота рекомендаций"}
            ],
            "business_metrics": [
                "Estimated Conversion Rate",
                "Expected Revenue",
                "Diversity Score"
            ],
            "online_metrics": [
                "CTR (Click-Through Rate)",
                "Conversion Rate",
                "Add-to-Cart Rate"
            ]
        }
    
    def _get_baseline_models(self):
        """Бейзлайн модели для сравнения"""
        return [
            {
                "name": "Popularity Baseline",
                "type": "Heuristic",
                "description": "Рекомендация самых популярных товаров"
            },
            {
                "name": "Item-Item CF",
                "type": "Collaborative Filtering",
                "description": "Коллаборативная фильтрация по товарам"
            },
            {
                "name": "Content-Based",
                "type": "Content Filtering", 
                "description": "Рекомендации на основе features товаров"
            }
        ]
    
    def _get_ab_testing_plan(self):
        """План A/B тестирования"""
        return {
            "test_duration": "2 недели",
            "traffic_allocation": {
                "control_group": 50,
                "treatment_group": 50
            },
            "primary_metric": "Conversion Rate",
            "secondary_metrics": [
                "CTR",
                "Average Order Value",
                "Bounce Rate"
            ],
            "sample_size_requirements": {
                "min_users_per_group": 10000,
                "statistical_power": 0.8,
                "significance_level": 0.05
            },
            "stopping_rules": [
                "Статистическая значимость p < 0.05",
                "Минимальная длительность 7 дней",
                "Мониторинг аномалий в метриках"
            ]
        }
    
    def generate_validation_code(self):
        """Генерация кода для валидации"""
        code_template = '''
# Валидационная стратегия - временной сплит
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Параметры временного сплита
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=30)  # 30 дней на тест

# Метрики для оценки
metrics = {
    'ndcg@10': [],
    'precision@5': [],
    'recall@20': []
}

for train_idx, test_idx in tscv.split(features):
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]
    
    # Обучение модели
    model.fit(X_train, y_train)
    
    # Предсказание и оценка
    predictions = model.predict_proba(X_test)
    metrics['ndcg@10'].append(calculate_ndcg(y_test, predictions, k=10))
    metrics['precision@5'].append(calculate_precision(y_test, predictions, k=5))
    metrics['recall@20'].append(calculate_recall(y_test, predictions, k=20))

print(f"Средние метрики по {n_splits} фолдам:")
for metric, values in metrics.items():
    print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
'''
        return code_template

# Использование
if __name__ == "__main__":
    # Создаем тестовые данные EDA для демонстрации
    eda_data = {
        "test_analysis": {"rows": 10000, "unique_users": 8826},
        "orders_analysis": {"unique_users": 3742, "rows": 4000}
    }
    
    validator = ValidationStrategy(eda_data)
    validation_plan = validator.create_validation_plan()
    
    # Сохранение плана валидации
    with open('validation_strategy.json', 'w', encoding='utf-8') as f:
        json.dump(validation_plan, f, ensure_ascii=False, indent=2)
    
    # Генерация кода валидации
    with open('validation_template.py', 'w', encoding='utf-8') as f:
        f.write(validator.generate_validation_code())
    
    print("Validation Strategy создана успешно!")
    print("Файлы созданы:")
    print("- validation_strategy.json")
    print("- validation_template.py")