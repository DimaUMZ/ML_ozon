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

    def expand_items_attributes(self, df):
        """Разбиение attributes в items на несколько признаков"""
        
        if df.empty:
            print("❌ No data to expand attributes")
            return pd.DataFrame()
        
        if 'attributes' not in df.columns:
            print("⚠️ Attributes expansion skipped - no 'attributes' column found")
            return df
        
        print(f"📊 Initial data for attributes expansion: {df.shape}")
        print(f"📋 Columns before expansion: {list(df.columns)}")
        
        def process_attributes(attr_list):
            if isinstance(attr_list, str):
                try:
                    if attr_list.startswith('['):
                        attr_list = ast.literal_eval(attr_list)
                    else:
                        attr_list = attr_list.replace('null', 'None')
                        attr_list = ast.literal_eval(attr_list)
                except:
                    return {}
            
            result = {}
            for attr in attr_list:
                name = attr.get('attribute_name')
                value = attr.get('attribute_value')
                if name:
                    if name in result:
                        if isinstance(result[name], list):
                            result[name].append(value)
                        else:
                            result[name] = [result[name], value]
                    else:
                        result[name] = value
            return result
        
        # Применяем функцию к каждой строке
        expanded_data = df['attributes'].apply(process_attributes)
        
        # Создаем DataFrame из словарей
        expanded_df = pd.json_normalize(expanded_data)
        
        # Объединяем с оригинальным DataFrame
        result_df = pd.concat([df.drop('attributes', axis=1), expanded_df], axis=1)
        
        # Удаляем признаки с слишком большим количеством пропусков (>30%)
        threshold = 0.3  # Порог: больше 30% пропусков
        missing_ratio = result_df.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if columns_to_drop:
            print(f"🗑️ Removing columns with >{threshold*100:.0f}% missing values: {columns_to_drop}")
            result_df = result_df.drop(columns=columns_to_drop)
        
        print(f"📊 After attributes expansion: {result_df.shape}")
        print(f"📋 Columns after expansion: {list(result_df.columns)}")
        
        # Статистика по пропускам
        if not result_df.empty:
            missing_stats = result_df.isnull().mean().sort_values(ascending=False)
            high_missing = missing_stats[missing_stats > 0]
            if not high_missing.empty:
                print("📈 Missing values statistics (columns with missing values):")
                for col, ratio in high_missing.items():
                    print(f"   - {col}: {ratio:.1%} missing")
        
        return result_df
