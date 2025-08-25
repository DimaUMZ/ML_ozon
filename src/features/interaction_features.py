import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import config

class InteractionFeatureEngineer:
    def __init__(self):
        self.params = config.get_feature_params()
    
    def calculate_time_based_features(self, df, user_col='user_id', timestamp_col='timestamp'):
        """Временные особенности взаимодействий"""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([user_col, timestamp_col])
        
        # Время с последнего взаимодействия
        df['time_since_last'] = df.groupby(user_col)[timestamp_col].diff().dt.total_seconds() / 3600
        df['time_since_last'] = df['time_since_last'].fillna(24)  # Первое взаимодействие
        
        # Длительность сессии (если есть session_id)
        if 'session_id' in df.columns:
            session_duration = df.groupby('session_id')[timestamp_col].agg(['min', 'max'])
            session_duration['duration_seconds'] = (
                session_duration['max'] - session_duration['min']
            ).dt.total_seconds()
            df = df.merge(session_duration[['duration_seconds']], on='session_id', how='left')
        
        return df
    
    def calculate_user_item_affinity(self, df, user_col='user_id', item_col='item_id',
                                   action_col='action', weight_dict={'view': 1, 'purchase': 5}):
        """Аффинити пользователя к товару"""
        # Взвешиваем действия
        df['action_weight'] = df[action_col].map(weight_dict).fillna(1)
        
        # Считаем взвешенный аффинити
        affinity = df.groupby([user_col, item_col])['action_weight'].sum().reset_index()
        affinity.rename(columns={'action_weight': 'user_item_affinity'}, inplace=True)
        
        return affinity
    
    def calculate_sequence_features(self, df, user_col='user_id', 
                                  item_col='item_id', timestamp_col='timestamp'):
        """Фичи последовательностей действий"""
        df = df.copy()
        df = df.sort_values([user_col, timestamp_col])
        
        # Позиция в сессии
        if 'session_id' in df.columns:
            df['action_position'] = df.groupby('session_id').cumcount() + 1
        
        # Разнообразие действий в сессии
        if 'session_id' in df.columns and 'action' in df.columns:
            session_diversity = df.groupby('session_id')['action'].nunique()
            df = df.merge(session_diversity.rename('session_diversity'), 
                         on='session_id', how='left')
        
        return df