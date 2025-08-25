# src/features/ozon_data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path

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
            # Ищем все parquet файлы
            parquet_files = list(data_dir.rglob("*.parquet"))
            parquet_files = [f for f in parquet_files if not f.name.startswith('._')]
            
            for file in parquet_files[:10]:  # Покажем первые 10
                print(f"   📄 {file.relative_to(self.root_dir)}")
            if len(parquet_files) > 10:
                print(f"   ... and {len(parquet_files) - 10} more")
        else:
            print("❌ Data directory does not exist")
    
    def load_interactions(self):
        """Загрузка или создание тестовых данных"""
        print("🔍 Looking for interactions data...")
        
        data_dir = self.root_dir / 'data'
        if data_dir.exists():
            # Ищем файлы взаимодействий
            possible_paths = [
                data_dir / 'ml_ozon_recsys_train_final_apparel_tracker_data',
                data_dir / 'ml_ozon_recsys_test_for_participants' / 'ml_ozon_recsys_train_final_apparel_tracker_data',
                data_dir / 'ml_ozon_recsys_train_final_apparel_orders_data'
            ]
            
            for path in possible_paths:
                if path.exists():
                    parquet_files = list(path.glob("*.parquet"))
                    parquet_files = [f for f in parquet_files if not f.name.startswith('._')]
                    
                    if parquet_files:
                        try:
                            df = pd.read_parquet(parquet_files[0])
                            print(f"✅ Loaded {len(df)} rows from {parquet_files[0]}")
                            return df
                        except Exception as e:
                            print(f"❌ Error reading {parquet_files[0]}: {e}")
        
        # Если данные не найдены, создаем тестовые
        print("📝 Creating sample interactions data...")
        return pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'item_id': [101, 102, 101, 103, 102, 104, 103, 105, 104, 106],
            'timestamp': pd.date_range('2025-01-01', periods=10, freq='H'),
            'action_type': ['view', 'purchase', 'view', 'view', 'purchase', 'view', 'cart', 'view', 'purchase', 'view']
        })