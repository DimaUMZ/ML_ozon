import numpy as np
from sklearn.metrics import ndcg_score

def calculate_ndcg(y_true, y_pred, k=10):
    """Calculate NDCG@k"""
    return ndcg_score(y_true, y_pred, k=k)

def calculate_precision(y_true, y_pred, k=5):
    """Calculate Precision@k"""
    top_k_pred = np.argsort(y_pred)[-k:]
    relevant = sum(y_true[top_k_pred])
    return relevant / k

def calculate_recall(y_true, y_pred, k=20):
    """Calculate Recall@k"""
    top_k_pred = np.argsort(y_pred)[-k:]
    relevant = sum(y_true[top_k_pred])
    total_relevant = sum(y_true)
    return relevant / total_relevant if total_relevant > 0 else 0