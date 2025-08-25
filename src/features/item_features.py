import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import config

class ItemFeatureEngineer:
    def __init__(self):
        self.params = config.get_feature_params()
    
    def calculate_item_popularity(self, df, item_col='item_id', 
                                action_col='action', timestamp_col='timestamp'):
        """Популярность товара в разные периоды"""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        item_popularity = {}
        current_time = pd.Timestamp.now()
        
        for window in self.params['time_windows']:
            # Популярность за последние N дней
            recent_data = df[
                df[timestamp_col] > (current_time - pd.Timedelta(days=window))
            ]
            popularity = recent_data[item_col].value_counts()
            item_popularity[f'item_popularity_{window}d'] = popularity
        
        return item_popularity
    
    def calculate_item_engagement_metrics(self, df, item_col='item_id', action_col='action'):
        """Метрики вовлеченности для товаров"""
        engagement_metrics = df.groupby([item_col, action_col]).size().unstack(fill_value=0)
        
        # CTR-like метрика
        if 'view' in engagement_metrics.columns and 'purchase' in engagement_metrics.columns:
            engagement_metrics['view_to_purchase_ratio'] = (
                engagement_metrics['purchase'] / engagement_metrics['view']
            ).replace([np.inf, -np.inf], 0).fillna(0)
        
        return engagement_metrics
    
    def calculate_item_temporal_patterns(self, df, item_col='item_id', timestamp_col='timestamp'):
        """Временные паттерны товаров"""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        
        # Популярность по часам (нормализованная)
        hour_popularity = df.groupby([item_col, 'hour']).size().unstack(fill_value=0)
        hour_popularity = hour_popularity.div(hour_popularity.sum(axis=1), axis=0)
        
        return hour_popularity