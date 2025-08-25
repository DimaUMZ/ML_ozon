from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..')
sys.path.append(src_path)

from evaluation.metrics import calculate_ndcg, calculate_precision, calculate_recall

def run_time_series_validation(model, features, targets, n_splits=3, test_size=30):
    """
    Запуск временной кросс-валидации
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    metrics = {
        'ndcg@10': [],
        'precision@5': [],
        'recall@20': []
    }

    for train_idx, test_idx in tscv.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]
        
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)[:, 1]  # Вероятности положительного класса
        
        metrics['ndcg@10'].append(calculate_ndcg(y_test, predictions, k=10))
        metrics['precision@5'].append(calculate_precision(y_test, predictions, k=5))
        metrics['recall@20'].append(calculate_recall(y_test, predictions, k=20))

    return metrics

def print_validation_results(metrics):
    """Печать результатов валидации"""
    print(f"Средние метрики по {len(metrics['ndcg@10'])} фолдам:")
    for metric, values in metrics.items():
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")