import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import config

class UserFeatureEngineer:
    def __init__(self):
        self.params = config.get_feature_params()
    
    def calculate_user_activity(self, df, user_col='user_id', timestamp_col='timestamp'):
        """Расчет активности пользователя в разные временные окна"""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        user_activity = {}
        for window in self.params['time_windows']:
            # Активность за последние N дней
            recent_interactions = df[
                df[timestamp_col] > (pd.Timestamp.now() - pd.Timedelta(days=window))
            ]
            activity = recent_interactions[user_col].value_counts()
            user_activity[f'user_activity_{window}d'] = activity
        
        return user_activity
    
    def calculate_user_purchase_metrics(self, df, user_col='user_id', 
                                      action_col='action', purchase_action='purchase'):
        """Метрики покупок пользователя"""
        purchases = df[df[action_col] == purchase_action]
        
        user_metrics = purchases.groupby(user_col).agg({
            'order_id': 'count',      # Количество покупок
            'price': ['sum', 'mean']  # Сумма и средний чек
        }).round(2)
        
        user_metrics.columns = ['purchase_count', 'total_spent', 'avg_order_value']
        return user_metrics
    
    def calculate_user_preferences(self, df, user_col='user_id', category_col='category'):
        """Предпочтения пользователя по категориям"""
        user_categories = df.groupby([user_col, category_col]).size().unstack(fill_value=0)
        user_preferences = user_categories.div(user_categories.sum(axis=1), axis=0)
        
        # Топ категория для каждого пользователя
        user_preferences['preferred_category'] = user_preferences.idxmax(axis=1)
        
        return user_preferences