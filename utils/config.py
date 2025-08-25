import json
import os

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            # Абсолютный путь к конфигу
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.config_path = os.path.join(base_dir, 'config', 'feature_config.json')
        else:
            self.config_path = config_path
        
        self.config = self._load_config()
    
    def _load_config(self):
        """Загрузка конфигурации из JSON файла"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using default parameters.")
            return {
                'feature_params': {
                    'time_windows': [1, 3, 7, 14, 30],
                    'min_user_interactions': 3
                }
            }
    
    def get(self, key, default=None):
        """Получение значения из конфига"""
        return self.config.get(key, default)
    
    def get_feature_params(self):
        """Получение параметров фичей"""
        return self.config.get('feature_params', {})

# Глобальный экземпляр конфига
config = Config()