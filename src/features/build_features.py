#!/usr/bin/env python3
"""
Главный скрипт для построения фичей рекомендательной системы OZON
"""

import pandas as pd
import numpy as np
import json
import os
import sys

# Добавляем корневую директорию в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

class DataLoader:
    def __init__(self, config_path='config/feature_config.json'):
        # Абсолютный путь к конфигу
        config_full_path = os.path.join(root_dir, config_path)
        self.config = self._load_config(config_full_path)
        self.data_paths = self.config.get('data_paths', {})
        
        print(f"📁 Config loaded from: {config_full_path}")
    
    def _load_config(self, config_path):
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Config file {config_path} not found. Using default paths.")
            return {
                'data_paths': {
                    'raw_interactions': 'data/ml_ozon_recsys_test_for_participants/ml_ozon_recsys_train_final_apparel_tracker_data/',
                    'raw_items': 'data/ml_ozon_recsys_test_for_participants/ml_ozon_recsys_train_final_apparel_items_data/',
                    'raw_orders': 'data/ml_ozon_recsys_test_for_participants/ml_ozon_recsys_train_final_apparel_orders_data/',
                    'processed_features': 'data/processed/features.parquet',
                    'processed_targets': 'data/processed/targets.parquet'
                }
            }
    
    def load_interactions(self):
        """Загрузка данных трекера с fallback на orders"""
        tracker_path = self.data_paths.get('raw_interactions', '')
        full_path = os.path.join(root_dir, tracker_path)
        
        result = self._load_parquet_data(full_path, "tracker interactions")
        
        # Если tracker не найден, используем orders как взаимодействия
        if result.empty:
            print("⚠️ Tracker data not found, trying to use orders as interactions...")
            orders_path = self.data_paths.get('raw_orders', '')
            orders_full_path = os.path.join(root_dir, orders_path)
            orders_data = self._load_parquet_data(orders_full_path, "orders")
            
            if not orders_data.empty:
                print("✅ Using orders data as interactions")
                # Переименовываем колонки для совместимости
                if 'user_id' in orders_data.columns and 'item_id' in orders_data.columns:
                    result = orders_data.copy()
                    # Добавляем фиктивное действие
                    result['action'] = 'purchase'
                    result['timestamp'] = pd.to_datetime('2025-01-01')  # Фиктивная дата
                else:
                    print("❌ Orders data doesn't have required columns")
        
        return result
    
    def load_items(self):
        """Загрузка данных о товарах"""
        items_path = self.data_paths.get('raw_items', '')
        full_path = os.path.join(root_dir, items_path)
        return self._load_parquet_data(full_path, "items")
    
    def load_orders(self):
        """Загрузка данных о заказах"""
        orders_path = self.data_paths.get('raw_orders', '')
        full_path = os.path.join(root_dir, orders_path)
        return self._load_parquet_data(full_path, "orders")
    
    def load_users(self):
        """Загрузка данных о пользователях"""
        print("ℹ️ User data not specified, returning empty DataFrame")
        return pd.DataFrame()
    
    def _load_parquet_data(self, data_path, data_type):
        """Универсальная загрузка parquet данных, игнорируя macOS файлы"""
        print(f"🔍 Looking for {data_type} in: {data_path}")
        
        # Проверка существования пути
        if not os.path.exists(data_path):
            print(f"❌ Path does not exist: {data_path}")
            
            # Рекурсивный поиск только настоящих parquet файлов
            print("🔍 Starting recursive search for parquet files...")
            parquet_files = self._find_parquet_files_recursive(root_dir)
            
            if parquet_files:
                print(f"✅ Found {len(parquet_files)} valid parquet files")
                
                # Ищем файлы по типу данных
                matching_files = []
                for file in parquet_files:
                    file_lower = file.lower()
                    # Игнорируем служебные файлы macOS и временные файлы
                    if ('__macosx' in file_lower or 
                        '/.' in file or 
                        file.startswith('._') or 
                        'cache' in file_lower):
                        continue
                    
                    # Ищем по ключевым словам в зависимости от типа данных
                    if data_type.lower() == "tracker interactions":
                        if any(keyword in file_lower for keyword in ['tracker', 'interaction', 'click', 'view', 'session']):
                            matching_files.append(file)
                    elif data_type.lower() == "items":
                        if any(keyword in file_lower for keyword in ['item', 'product', 'catalog', 'goods']):
                            matching_files.append(file)
                    elif data_type.lower() == "orders":
                        if any(keyword in file_lower for keyword in ['order', 'purchase', 'buy', 'transaction']):
                            matching_files.append(file)
                
                if matching_files:
                    print(f"📁 Found {len(matching_files)} matching files for {data_type}")
                    for i, file in enumerate(matching_files[:5]):  # Покажем первые 5
                        print(f"   {i+1}. {file}")
                    
                    # Пробуем загрузить файлы по порядку
                    for file in matching_files:
                        print(f"📁 Trying to load: {file}")
                        try:
                            df = pd.read_parquet(file)
                            print(f"✅ Successfully loaded {len(df)} rows from {file}")
                            return df
                        except Exception as e:
                            print(f"❌ Failed to load {file}: {e}")
                            continue
                
                print(f"⚠️ No matching files found for {data_type}")
            else:
                print("❌ No parquet files found anywhere in the project")
            
            return pd.DataFrame()
        
        # Если путь существует
        if os.path.isdir(data_path):
            # Ищем только настоящие parquet файлы (игнорируем macOS)
            files = [f for f in os.listdir(data_path) 
                    if f.endswith('.parquet') and not f.startswith('._')]
            
            if files:
                print(f"✅ Found {len(files)} parquet files in directory")
                for file in files:
                    print(f"   - {file}")
                
                # Пробуем загрузить по порядку
                for file in files:
                    file_path = os.path.join(data_path, file)
                    print(f"📁 Trying to load: {file_path}")
                    try:
                        df = pd.read_parquet(file_path)
                        print(f"✅ Successfully loaded {len(df)} rows from {file}")
                        return df
                    except Exception as e:
                        print(f"❌ Failed to load {file_path}: {e}")
                        continue
                
                print(f"❌ All files in {data_path} failed to load")
            else:
                print(f"❌ No valid parquet files found in: {data_path}")
        
        elif os.path.isfile(data_path) and data_path.endswith('.parquet'):
            # Проверяем что это не macOS файл
            if os.path.basename(data_path).startswith('._'):
                print(f"❌ Skipping macOS metadata file: {data_path}")
                return pd.DataFrame()
            
            try:
                df = pd.read_parquet(data_path)
                print(f"✅ Loaded {len(df)} {data_type} from file")
                return df
            except Exception as e:
                print(f"❌ Error reading {data_path}: {e}")
        
        return pd.DataFrame()
    
    def _find_parquet_files_recursive(self, search_path):
        """Рекурсивный поиск parquet файлов, игнорируя служебные файлы macOS"""
        parquet_files = []
        
        if not os.path.exists(search_path):
            return parquet_files
        
        try:
            for root, dirs, files in os.walk(search_path):
                # Пропускаем скрытые директории и macOS служебные
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
                
                for file in files:
                    if (file.endswith('.parquet') and 
                        not file.startswith('._') and  # Игнорируем macOS метаданные
                        not root.endswith('/__MACOSX')):  # Игнорируем macOS папки
                        
                        full_path = os.path.join(root, file)
                        parquet_files.append(full_path)
                        
        except Exception as e:
            print(f"⚠️ Error during search: {e}")
        
        # Сортируем по пути для удобства просмотра
        parquet_files.sort()
        return parquet_files
    
    def merge_datasets(self, interactions, items, orders, users):
        """Объединение datasets"""
        if interactions.empty:
            print("❌ No interactions data to merge")
            return pd.DataFrame()
        
        merged = interactions.copy()
        print(f"📊 Initial interactions: {merged.shape}")
        print(f"📋 Interactions columns: {list(merged.columns)}")
        
        # Мердж с товарами
        if not items.empty and 'item_id' in interactions.columns and 'item_id' in items.columns:
            merged = merged.merge(
                items, 
                on='item_id', 
                how='left',
                suffixes=('', '_item')
            )
            print(f"📊 Merged with items: {merged.shape}")
        else:
            print("⚠️ Items merge skipped")
        
        # Мердж с заказами (если есть общие колонки)
        if not orders.empty:
            # Проверяем возможные колонки для мерджа
            common_columns = set(merged.columns) & set(orders.columns)
            print(f"🔍 Common columns with orders: {common_columns}")
            
            if 'order_id' in common_columns:
                merged = merged.merge(
                    orders,
                    on='order_id',
                    how='left',
                    suffixes=('', '_order')
                )
                print(f"📊 Merged with orders: {merged.shape}")
            elif 'user_id' in common_columns and 'item_id' in common_columns:
                # Альтернативный мердж по user_id + item_id
                merged = merged.merge(
                    orders,
                    on=['user_id', 'item_id'],
                    how='left',
                    suffixes=('', '_order')
                )
                print(f"📊 Merged with orders by user+item: {merged.shape}")
            else:
                print("⚠️ Orders merge skipped - no common columns")
        
        # Мердж с пользователями
        if not users.empty and 'user_id' in merged.columns and 'user_id' in users.columns:
            merged = merged.merge(
                users,
                on='user_id',
                how='left',
                suffixes=('', '_user')
            )
            print(f"📊 Merged with users: {merged.shape}")
        
        print(f"📋 Final columns: {list(merged.columns)}")
        return merged
    
    def save_processed_data(self, features, targets=None):
        """Сохранение обработанных данных"""
        try:
            # Создаем директории если не существуют
            processed_features_path = os.path.join(root_dir, self.data_paths['processed_features'])
            os.makedirs(os.path.dirname(processed_features_path), exist_ok=True)
            
            # Удаляем проблемные колонки перед сохранением
            columns_to_drop = ['fclip_embed', 'attributes']  # Колонки которые вызывают ошибки
            features_to_save = features.drop(columns=[col for col in columns_to_drop 
                                                    if col in features.columns])
            
            print(f"📊 Saving {features_to_save.shape[1]} columns (removed problematic columns)")
            
            # Сохраняем фичи
            features_to_save.to_parquet(processed_features_path)
            print(f"💾 Features saved to: {processed_features_path}")
            
            # Сохраняем таргеты если есть
            if targets is not None:
                processed_targets_path = os.path.join(root_dir, self.data_paths['processed_targets'])
                # Преобразуем Series в DataFrame для сохранения
                targets_df = pd.DataFrame({'target': targets})
                targets_df.to_parquet(processed_targets_path)
                print(f"💾 Targets saved to: {processed_targets_path}")
                print(f"   Targets shape: {targets_df.shape}")
                
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            import traceback
            traceback.print_exc()

class FeatureEngineer:
    """Простой инженер фичей для демонстрации"""
    
    @staticmethod
    def create_basic_features(df):
        """Создание базовых фичей"""
        if df.empty:
            return df
        
        df = df.copy()
        print("🛠️ Creating basic features...")
        
        # Временные фичи
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                print("✅ Temporal features created")
            except Exception as e:
                print(f"❌ Error creating temporal features: {e}")
        
        # User features
        if 'user_id' in df.columns:
            user_counts = df['user_id'].value_counts()
            df['user_activity_count'] = df['user_id'].map(user_counts)
            print("✅ User features created")
        
        # Item features
        if 'item_id' in df.columns:
            item_counts = df['item_id'].value_counts()
            df['item_popularity'] = df['item_id'].map(item_counts)
            print("✅ Item features created")
        
        print(f"🛠️ Final features shape: {df.shape}")
        return df
    
    @staticmethod
    def create_targets(df, action_column='action_type', positive_actions=['buy', 'purchase', 'cart', 'order']):
        """Создание целевой переменной"""
        if df.empty or action_column not in df.columns:
            print(f"⚠️ Cannot create targets - no '{action_column}' column")
            print(f"   Available columns: {list(df.columns)}")
            return None
        
        # Посмотрим какие действия есть в данных
        unique_actions = df[action_column].astype(str).unique()
        print(f"🔍 Unique actions in data: {unique_actions}")
        
        # Приводим к нижнему регистру для сравнения
        actions = df[action_column].astype(str).str.lower()
        targets = actions.isin([a.lower() for a in positive_actions]).astype(int)
        
        positive_count = targets.sum()
        total_count = len(targets)
        
        print(f"🎯 Targets created from '{action_column}': {positive_count}/{total_count} positive samples ({positive_count/total_count:.1%})")
        
        # Если нет позитивных samples, попробуем другие варианты
        if positive_count == 0:
            print("⚠️ No positive samples found. Trying alternative actions...")
            # Попробуем найти любые не-view действия
            non_view_actions = [action for action in unique_actions 
                            if 'view' not in str(action).lower() and 'click' not in str(action).lower()]
            if non_view_actions:
                print(f"🔍 Trying alternative actions: {non_view_actions[:3]}")
                targets = actions.isin([str(a).lower() for a in non_view_actions]).astype(int)
                positive_count = targets.sum()
                print(f"🎯 Alternative targets: {positive_count}/{total_count} positive samples ({positive_count/total_count:.1%})")
        
        return targets

def main():
    """Главная функция для построения фичей"""
    print("=" * 50)
    print("🚀 STARTING OZON FEATURE ENGINEERING PIPELINE")
    print("=" * 50)
    
    
    try:
        # Инициализация
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer()
        
        # 1. Загрузка данных
        print("\n1. 📥 LOADING DATA")
        print("-" * 30)
        
        interactions = data_loader.load_interactions()
        items = data_loader.load_items()
        orders = data_loader.load_orders()
        users = data_loader.load_users()
        print(f"🔍 Actions in interactions: {interactions['action_type'].unique()}")
        print(f"📊 Action counts:")
        print(interactions['action_type'].value_counts())
        
        print(f"   📊 Interactions: {interactions.shape}")
        print(f"   📦 Items: {items.shape}")
        print(f"   🛒 Orders: {orders.shape}")
        print(f"   👥 Users: {users.shape}")
        
        # 2. Объединение данных
        print("\n2. 🔄 MERGING DATASETS")
        print("-" * 30)
        
        merged_data = data_loader.merge_datasets(interactions, items, orders, users)
        
        if merged_data.empty:
            print("❌ No data to process. Exiting.")
            return None
        
        print(f"   📊 Merged data: {merged_data.shape}")
        print(f"   📋 Columns: {list(merged_data.columns)}")
        
        # 3. Построение фичей
        print("\n3. 🛠️ FEATURE ENGINEERING")
        print("-" * 30)
        
        features = feature_engineer.create_basic_features(merged_data)
        
        # 4. Создание таргетов
        print("\n4. 🎯 CREATING TARGETS")
        print("-" * 30)
        
        targets = feature_engineer.create_targets(features)
        
        # 5. Сохранение результатов
        print("\n5. 💾 SAVING RESULTS")
        print("-" * 30)
        
        data_loader.save_processed_data(features, targets)
        
        print("\n" + "=" * 50)
        print("✅ FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        return features, targets
        
    except Exception as e:
        print(f"\n❌ ERROR IN PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()