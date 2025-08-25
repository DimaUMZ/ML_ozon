import pandas as pd
import numpy as np
import json

class FeatureUnion:
    def __init__(self, config_path='../../config/feature_config.json'):
        self.config = self._load_config(config_path)
        self.params = self.config.get('feature_params', {})
    
    def _load_config(self, config_path):
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default parameters.")
            return {
                'feature_params': {
                    'time_windows': [1, 3, 7, 14, 30],
                    'min_user_interactions': 3
                }
            }
    
    def build_all_features(self, df):
        """Построение всех фичей с использованием параметров из конфига"""
        if df.empty:
            print("Empty DataFrame, skipping feature engineering")
            return df
        
        print("Building basic features...")
        
        # Базовые временные фичи
        df = self._extract_temporal_features(df)
        
        # User features с параметрами из конфига
        df = self._build_user_features(df)
        
        # Item features  
        df = self._build_item_features(df)
        
        # Interaction features
        df = self._build_interaction_features(df)
        
        return df
    
    def _extract_temporal_features(self, df):
        """Извлечение временных фичей"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df
    
    def _build_user_features(self, df):
        """User features с параметрами из конфига"""
        time_windows = self.params.get('time_windows', [1, 3, 7, 14, 30])
        
        if 'user_id' in df.columns and 'timestamp' in df.columns:
            # Активность пользователя в разные окна
            for window in time_windows:
                df[f'user_activity_{window}d'] = self._calculate_user_activity(df, window)
        
        return df
    
    def _calculate_user_activity(self, df, window_days):
        """Расчет активности пользователя за указанное окно"""
        # Упрощенная реализация
        return df.groupby('user_id')['user_id'].transform('count')  # Заглушка
    
    def _build_item_features(self, df):
        """Базовые item features"""
        if 'item_id' in df.columns:
            # Популярность товара
            item_popularity = df['item_id'].value_counts().rename('item_popularity_total')
            df = df.merge(item_popularity.rename('item_popularity'), 
                         left_on='item_id', right_index=True, how='left')
        return df
    
    def _build_interaction_features(self, df):
        """Базовые interaction features"""
        # Время с последнего действия (если есть timestamp и user_id)
        if 'timestamp' in df.columns and 'user_id' in df.columns:
            df = df.sort_values(['user_id', 'timestamp'])
            df['time_since_last'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
        
        return df