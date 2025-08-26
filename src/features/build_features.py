#!/usr/bin/env python3
"""
Главный скрипт для построения фичей рекомендательной системы OZON
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

# Убираем импорт через src, импортируем напрямую
import importlib.util

# Функция для импорта модуля по пути
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec) # type: ignore
    spec.loader.exec_module(module) # type: ignore
    return module

# Импортируем наши модули
ozon_loader_path = root_dir / 'src' / 'features' / 'ozon_data_loader.py'
feature_engineer_path = root_dir / 'src' / 'features' / 'enhanced_feature_engineer.py'

# Проверяем существование файлов
print(f"🔍 Checking loader path: {ozon_loader_path}")
print(f"🔍 Checking engineer path: {feature_engineer_path}")

if ozon_loader_path.exists() and feature_engineer_path.exists():
    OzonDataLoader = import_module_from_path('ozon_data_loader', ozon_loader_path).OzonDataLoader
    EnhancedFeatureEngineer = import_module_from_path('enhanced_feature_engineer', feature_engineer_path).EnhancedFeatureEngineer
    print("✅ Modules imported successfully")
else:
    print("❌ Module files not found, creating simple implementation...")
    
    # Простая реализация если файлы не найдены
    class OzonDataLoader:
        def __init__(self, config_path='config/feature_config.json'):
            self.root_dir = Path(__file__).parent.parent.parent
            print(f"📁 Project root: {self.root_dir}")
        
        def explore_data_structure(self):
            """Исследование структуры данных для отладки"""
            print("🔍 Exploring data directory structure...")
            data_dir = self.root_dir / 'data'
            if data_dir.exists():
                print("✅ Data directory exists")
                for item in data_dir.rglob("*.parquet"):
                    if not item.name.startswith('._'):
                        print(f"   📄 {item.relative_to(self.root_dir)}")
            else:
                print("❌ Data directory does not exist")
        
        def load_interactions(self):
            """Создаем тестовые данные"""
            print("📝 Creating sample interactions data...")
            return pd.DataFrame({
                'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                'item_id': [101, 102, 101, 103, 102, 104, 103, 105, 104, 106],
                'timestamp': pd.date_range('2025-01-01', periods=10, freq='H'),
                'action_type': ['view', 'purchase', 'view', 'view', 'purchase', 'view', 'cart', 'view', 'purchase', 'view']
            })
    
    class EnhancedFeatureEngineer:
        def __init__(self, config=None):
            self.config = config or {}
            self.time_windows = self.config.get('time_windows', [1, 3, 7, 14, 30])
        
        def create_enhanced_features(self, df):
            """Создание расширенных фичей"""
            if df.empty:
                return df
            
            df = df.copy()
            print("🛠️ Creating enhanced features...")
            
            # Базовые временные фичи
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # User features
            if 'user_id' in df.columns:
                user_stats = df.groupby('user_id').agg({
                    'timestamp': ['count', 'min', 'max']
                }).reset_index()
                user_stats.columns = ['user_id', 'user_total_events', 'first_activity', 'last_activity']
                df = df.merge(user_stats, on='user_id', how='left')
            
            # Item features
            if 'item_id' in df.columns:
                item_stats = df.groupby('item_id').agg({
                    'timestamp': 'count',
                    'user_id': 'nunique'
                }).reset_index()
                item_stats.columns = ['item_id', 'item_popularity', 'unique_users']
                df = df.merge(item_stats, on='item_id', how='left')
            
            return df
        
        def create_improved_targets(self, df):
            """Создание таргетов"""
            if 'action_type' not in df.columns:
                return None
            
            # Простое определение таргетов
            positive_actions = ['purchase', 'buy', 'cart', 'order']
            df['action_lower'] = df['action_type'].astype(str).str.lower()
            targets = df['action_lower'].isin(positive_actions).astype(int)
            
            print(f"🎯 Targets: {targets.sum()}/{len(targets)} positive ({targets.mean():.2%})")
            return targets

        def optimize_datasets_dtypes(self, df, verbose=True):
            """Оптимизироване типов данных для уменьшения размера в памяти"""
            if df.empty:
                print("❌ DataFrame is empty - skipping optimization")
                return df
        
            optimized_df = df.copy()
            total_memory_before = optimized_df.memory_usage(deep=True).sum()
            changes_made = 0
        
            print(f"🧮 Initial memory usage: {total_memory_before / 1024 ** 2:.2f} MB")
            print(f"📊 Initial shape: {optimized_df.shape}")
        
            for column in optimized_df.columns:
                current_dtype = optimized_df[column].dtype
                memory_before = optimized_df[column].memory_usage(deep=True)
            
                # Пропускаем datetime колонки
                if pd.api.types.is_datetime64_any_dtype(current_dtype):
                    if verbose:
                        print(f"   ⏰ {column}: {current_dtype} (datetime - skipped)")
                    continue
            
                # Пропускаем сложные типы данных
                if current_dtype == object and optimized_df[column].apply(
                    lambda x: isinstance(x, (list, dict, np.ndarray))).any():
                    if verbose:
                        print(f"   🔄 {column}: complex type (list/dict/array - skipped)")
                    continue
            
                new_dtype = current_dtype
             
                # Оптимизация числовых колонок
                if pd.api.types.is_numeric_dtype(current_dtype):
                    col_min = optimized_df[column].min()
                    col_max = optimized_df[column].max()
                
                    if pd.api.types.is_integer_dtype(current_dtype):
                        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                            new_dtype = 'int8'
                        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                            new_dtype = 'int16'
                        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                            new_dtype = 'int32'
            
                # Оптимизация строковых колонок в категории
                elif current_dtype == 'object':
                    unique_ratio = optimized_df[column].nunique() / len(optimized_df[column])
                    if unique_ratio < 0.5:  # Меньше 50% уникальных значений
                        new_dtype = 'category'
            
                # Применяем изменение типа если нужно
                if str(new_dtype) != str(current_dtype):
                    try:
                        original_values = optimized_df[column].copy()
                        optimized_df[column] = optimized_df[column].astype(new_dtype)
                    
                        # Проверка целостности данных
                        if pd.api.types.is_numeric_dtype(current_dtype):
                            if not np.array_equal(original_values, optimized_df[column], equal_nan=True):
                                optimized_df[column] = original_values
                                if verbose:
                                    print(f"   ⚠️ {column}: data integrity check failed - reverted")
                                continue
                    
                        memory_after = optimized_df[column].memory_usage(deep=True)
                        memory_saved = memory_before - memory_after
                        changes_made += 1
                    
                        if verbose:
                            print(f"   ✅ {column}: {current_dtype} → {new_dtype} "
                                f"(saved: {memory_saved / 1024:.1f} KB)")
                        
                    except (ValueError, TypeError) as e:
                        if verbose:
                            print(f"   ❌ {column}: conversion failed {current_dtype} → {new_dtype} ({str(e)})")
        
            # Вывод итоговой статистики
            total_memory_after = optimized_df.memory_usage(deep=True).sum()
            total_memory_saved = total_memory_before - total_memory_after
        
            print(f"📈 Final memory usage: {total_memory_after / 1024 ** 2:.2f} MB")
            print(f"💾 Total memory saved: {total_memory_saved / 1024 ** 2:.2f} MB")
            print(f"🔧 Columns optimized: {changes_made}/{len(df.columns)}")
        
            return optimized_df

def explore_directory():
    """Исследование структуры директорий для отладки"""
    print("🔍 Exploring directory structure...")
    
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / 'data'
    print(f"📁 Root: {root_dir}")
    print(f"📁 Data: {data_dir}")
    
    if data_dir.exists():
        print("✅ Data directory exists")
        parquet_files = list(data_dir.rglob("*.parquet"))
        for file in parquet_files[:10]:  # Покажем первые 10 файлов
            if not file.name.startswith('._'):
                print(f"   📄 {file.relative_to(root_dir)}")
        if len(parquet_files) > 10:
            print(f"   ... and {len(parquet_files) - 10} more")
    else:
        print("❌ Data directory does not exist")

def main():
    """Главная функция для построения фичей"""
    print("=" * 60)
    print("🚀 STARTING OZON FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    try:
        # 0. Исследуем структуру директорий
        print("\n0. 🔍 DIRECTORY EXPLORATION")
        print("-" * 40)
        explore_directory()
        
        # 1. Инициализация
        print("\n1. 📥 INITIALIZATION")
        print("-" * 40)
        
        data_loader = OzonDataLoader()
        
        # 2. Загрузка данных
        print("\n2. 📥 LOADING DATA")
        print("-" * 40)
        
        data_loader.explore_data_structure()
        interactions = data_loader.load_interactions()
        
        print(f"\n📊 Interactions shape: {interactions.shape}")
        print(f"📋 Interactions columns: {list(interactions.columns)}")
        
        if not interactions.empty and 'action_type' in interactions.columns:
            print(f"🔍 Action types: {interactions['action_type'].unique()}")
        
        # 3. Загрузка конфига
        print("\n3. ⚙️ LOADING CONFIG")
        print("-" * 40)
        
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / 'config' / 'feature_config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ Config loaded successfully")
        else:
            print("⚠️ Config file not found, using defaults")
            config = {
                'feature_params': {
                    'time_windows': [1, 3, 7, 14, 30],
                    'min_user_interactions': 3
                }
            }
        
        # 4. Feature Engineering
        print("\n4. 🛠️ FEATURE ENGINEERING")
        print("-" * 40)
        
        feature_engineer = EnhancedFeatureEngineer(config.get('feature_params', {}))
        features = feature_engineer.create_enhanced_features(interactions)
        
        print(f"📊 Enhanced features shape: {features.shape}")
        print(f"📋 New columns: {[col for col in features.columns if col not in interactions.columns]}")
        
        # 5. Создание таргетов
        print("\n5. 🎯 CREATING TARGETS")
        print("-" * 40)
        
        targets = feature_engineer.create_improved_targets(features)
        
        if targets is None:
            print("❌ Failed to create targets")
            return None, None
        
        # 6. Сохранение результатов
        print("\n6. 💾 SAVING RESULTS")
        print("-" * 40)
        
        root_dir = Path(__file__).parent.parent.parent
        processed_dir = root_dir / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        features_path = processed_dir / 'features.parquet'
        targets_path = processed_dir / 'targets.parquet'
        
        features.to_parquet(features_path)
        pd.DataFrame({'target': targets}).to_parquet(targets_path)
        
        print(f"✅ Features saved to: {features_path}")
        print(f"✅ Targets saved to: {targets_path}")
        print(f"📊 Final features shape: {features.shape}")
        print(f"🎯 Target distribution: {targets.sum()}/{len(targets)} positive ({targets.mean():.2%})")
        
        print("\n" + "=" * 60)
        print("✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return features, targets
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()
