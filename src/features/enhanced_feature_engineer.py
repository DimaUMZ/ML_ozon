import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EnhancedFeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.time_windows = config.get('time_windows', [1, 3, 7, 14, 30])
    
    def create_enhanced_features(self, df):
        """Создание расширенных фичей с временными окнами"""
        if df.empty:
            return df
        
        df = df.copy()
        print("🛠️ Creating enhanced features with time windows...")
        
        # Временные фичи
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            current_time = df['timestamp'].max()
            
            # Базовые временные фичи
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # User features с временными окнами
            if 'user_id' in df.columns:
                df = self._add_user_time_features(df, current_time)
            
            # Item features с временными окнами
            if 'item_id' in df.columns:
                df = self._add_item_time_features(df, current_time)
        
        return df
    
    def _add_user_time_features(self, df, current_time):
        """Добавление user features с временными окнами"""
        user_features = []
        
        for user_id in df['user_id'].unique():
            user_mask = df['user_id'] == user_id
            user_data = df[user_mask]
            
            user_stats = {
                'user_id': user_id,
                'user_total_events': len(user_data)
            }
            
            # Активность по временным окнам
            for window in self.time_windows:
                window_start = current_time - timedelta(days=window)
                window_events = user_data[user_data['timestamp'] > window_start]
                user_stats[f'user_activity_{window}d'] = len(window_events)
            
            user_features.append(user_stats)
        
        user_df = pd.DataFrame(user_features)
        return df.merge(user_df, on='user_id', how='left')
    
    def _add_item_time_features(self, df, current_time):
        """Добавление item features с временными окнами"""
        item_features = []
        
        for item_id in df['item_id'].unique():
            item_mask = df['item_id'] == item_id
            item_data = df[item_mask]
            
            item_stats = {
                'item_id': item_id,
                'item_total_views': len(item_data)
            }
            
            # Популярность по временным окнам
            for window in self.time_windows:
                window_start = current_time - timedelta(days=window)
                window_views = item_data[item_data['timestamp'] > window_start]
                item_stats[f'item_popularity_{window}d'] = len(window_views)
            
            # Конверсия (если есть данные о покупках)
            if 'action_type' in df.columns:
                purchases = item_data[item_data['action_type'].str.contains('buy|purchase|order', case=False, na=False)]
                item_stats['item_purchase_count'] = len(purchases)
                item_stats['item_conversion_rate'] = len(purchases) / len(item_data) if len(item_data) > 0 else 0
            
            item_features.append(item_stats)
        
        item_df = pd.DataFrame(item_features)
        return df.merge(item_df, on='item_id', how='left')
    
    def create_improved_targets(self, df):
        """Улучшенное создание таргетов"""
        if 'action_type' not in df.columns:
            return None
        
        # Более точное определение позитивных действий
        positive_keywords = ['buy', 'purchase', 'order', 'cart_add', 'to_cart', 'favorite']
        negative_keywords = ['view', 'click', 'page_view', 'remove', 'unfavorite']
        
        df['action_lower'] = df['action_type'].astype(str).str.lower()
        
        # Позитивные действия
        positive_mask = df['action_lower'].str.contains('|'.join(positive_keywords), na=False)
        
        # Негативные действия (только просмотры без конверсии)
        negative_mask = df['action_lower'].str.contains('|'.join(negative_keywords), na=False)
        
        # Создаем таргет: 1 для покупок, 0 для просмотров
        targets = positive_mask.astype(int)
        
        # Для пользователей с покупками, помечаем все их просмотры как 0?
        # Альтернативно: использовать только завершенные сессии
        
        print(f"🎯 Targets: {targets.sum()}/{len(targets)} positive ({targets.mean():.3%})")
        return targets