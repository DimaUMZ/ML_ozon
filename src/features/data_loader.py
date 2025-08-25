import pandas as pd
import numpy as np
import json
import os

class DataLoader:
    def __init__(self, config_path='config/feature_config.json', root_dir=None):
        # Сохраняем root_dir
        self.root_dir = root_dir or os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Абсолютный путь к конфигу
        config_full_path = os.path.join(self.root_dir, config_path)
        self.config = self._load_config(config_full_path)
        self.data_paths = self.config.get('data_paths', {})
        
        # Для отладки
        print(f"📁 Root directory: {self.root_dir}")
        print(f"📁 Config data paths: {self.data_paths}")
    
    def _load_config(self, config_path):
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default paths.")
            return {
                'data_paths': {
                    'raw_interactions': 'data/ml_ozon_recsys_test_for_participants/ml_ozon_recsys_train_final_apparel_tracker_data/',
                    'raw_items': 'data/ml_ozon_recsys_test_for_participants/ml_ozon_recsys_train_final_apparel_items_data/',
                    'raw_orders': 'data/ml_ozon_recsys_test_for_participants/ml_ozon_recsys_train_final_apparel_orders_data/'
                }
            }
    
    def load_interactions(self):
        """Загрузка данных трекера"""
        tracker_path = self.data_paths.get('raw_interactions', '')
        full_path = os.path.join(self.root_dir, tracker_path)  # Добавить self.root_dir
        return self._load_parquet_data(full_path, "tracker interactions")

    def load_items(self):
        """Загрузка данных о товарах"""
        items_path = self.data_paths.get('raw_items', '')
        full_path = os.path.join(self.root_dir, items_path)  # Добавить self.root_dir
        return self._load_parquet_data(full_path, "items")
    
    def load_users(self):
        """Загрузка данных о пользователях"""
        # Если нет отдельных users данных, возвращаем пустой DataFrame
        print("User data not specified in config, returning empty DataFrame")
        return pd.DataFrame()
    
    def _load_parquet_data(self, data_path, data_type):
        """Универсальная загрузка parquet данных с рекурсивным поиском"""
        print(f"Looking for {data_type} in: {data_path}")
        
        # Проверка существования пути
        if not os.path.exists(data_path):
            print(f"❌ Path does not exist: {data_path}")
            
            # Рекурсивный поиск parquet файлов во всем проекте
            print("🔍 Starting recursive search for parquet files...")
            parquet_files = self._find_parquet_files_recursive('/home/dima/ozon')
            
            if parquet_files:
                print("✅ Found parquet files:")
                for file in parquet_files:
                    print(f"   - {file}")
                
                # Используем первый найденный файл для данного типа данных
                if parquet_files:
                    first_file = parquet_files[0]
                    print(f"📁 Using first found file: {first_file}")
                    try:
                        df = pd.read_parquet(first_file)
                        print(f"✅ Loaded {len(df)} {data_type} from {first_file}")
                        return df
                    except Exception as e:
                        print(f"❌ Error reading {first_file}: {e}")
            else:
                print("❌ No parquet files found anywhere in the project")
            
            return pd.DataFrame()
        
        # Если путь существует, проверяем папку или файл
        if os.path.isdir(data_path):
            # Рекурсивный поиск в указанной папке
            parquet_files = self._find_parquet_files_recursive(data_path)
            
            if parquet_files:
                print(f"✅ Found {len(parquet_files)} parquet files:")
                for file in parquet_files[:3]:  # Покажем первые 3 файла
                    print(f"   - {file}")
                if len(parquet_files) > 3:
                    print(f"   ... and {len(parquet_files) - 3} more")
                
                # Используем первый файл
                first_file = parquet_files[0]
                try:
                    df = pd.read_parquet(first_file)
                    print(f"✅ Loaded {len(df)} {data_type} from {first_file}")
                    return df
                except Exception as e:
                    print(f"❌ Error reading {first_file}: {e}")
            else:
                print(f"❌ No parquet files found in: {data_path}")
                # Рекурсивный поиск во всем проекте
                print("🔍 Searching in entire project...")
                all_parquet_files = self._find_parquet_files_recursive('/home/dima/ozon')
                if all_parquet_files:
                    print("Found parquet files in project:")
                    for file in all_parquet_files:
                        print(f"   - {file}")
        
        elif os.path.isfile(data_path) and data_path.endswith('.parquet'):
            # Это parquet файл
            try:
                df = pd.read_parquet(data_path)
                print(f"✅ Loaded {len(df)} {data_type} from file")
                return df
            except Exception as e:
                print(f"❌ Error reading {data_path}: {e}")
        
        return pd.DataFrame()

    def _find_parquet_files_recursive(self, search_path):
        """Рекурсивный поиск всех parquet файлов в указанной папке"""
        parquet_files = []
        
        if not os.path.exists(search_path):
            return parquet_files
        
        try:
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.parquet'):
                        full_path = os.path.join(root, file)
                        parquet_files.append(full_path)
        except Exception as e:
            print(f"❌ Error during recursive search: {e}")
        
        return parquet_files
        
    def merge_datasets(self, interactions, items, users):
        """Объединение datasets"""
        if interactions.empty:
            print("No interactions data to merge")
            return pd.DataFrame()
        
        # Мердж с товарами
        if not items.empty and 'item_id' in interactions.columns and 'item_id' in items.columns:
            merged = interactions.merge(
                items, 
                on='item_id', 
                how='left',
                suffixes=('', '_item')
            )
            print(f"Merged with items: {merged.shape}")
        else:
            merged = interactions
            print("Items merge skipped - no items data or missing item_id column")
        
        # Мердж с пользователями (если есть данные)
        if not users.empty and 'user_id' in merged.columns and 'user_id' in users.columns:
            merged = merged.merge(
                users,
                on='user_id',
                how='left',
                suffixes=('', '_user')
            )
            print(f"Merged with users: {merged.shape}")
        
        return merged