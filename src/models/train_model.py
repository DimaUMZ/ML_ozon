#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import pickle
import os
import sys

# Добавляем корневую директорию в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.label_encoders = {}
        
    def load_data(self, features_path, targets_path):
        """Загрузка подготовленных данных"""
        print("📥 Loading processed data...")
        
        features = pd.read_parquet(features_path)
        targets = pd.read_parquet(targets_path)
        
        print(f"📊 Features shape: {features.shape}")
        print(f"🎯 Targets shape: {targets.shape}")
        print(f"📋 Features columns: {list(features.columns)}")
        print(f"📋 Features dtypes:\n{features.dtypes}")
        
        return features, targets['target']  # Извлекаем Series из DataFrame

    def prepare_data(self, features, targets, test_size=0.2, random_state=42):
        """Улучшенная подготовка данных для обучения"""
        print("🛠️ Preparing data for training...")
        
        # Создаем копию чтобы не изменять оригинальные данные
        features_processed = features.copy()
        
        # 1. Удаляем проблемные колонки
        columns_to_drop = [
            'action_type', 'action_widget', 'last_status', 'itemname',
            'action_lower', 'timestamp', 'created_timestamp', 
            'last_status_timestamp', 'date', 'created_date'
        ]
        
        for col in columns_to_drop:
            if col in features_processed.columns:
                print(f"   Dropping column: {col}")
                features_processed = features_processed.drop(columns=[col])
        
        # 2. Обрабатываем категориальные переменные
        categorical_columns = features_processed.select_dtypes(include=['object', 'category']).columns
        print(f"   Categorical columns: {list(categorical_columns)}")
        
        for col in categorical_columns:
            if col in features_processed.columns:
                print(f"   Encoding categorical column: {col}")
                # Заменяем пропуски
                features_processed[col] = features_processed[col].fillna('unknown')
                
                # Label Encoding для категориальных переменных
                le = LabelEncoder()
                features_processed[col] = le.fit_transform(features_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # 3. Обрабатываем числовые переменные
        numeric_columns = features_processed.select_dtypes(include=[np.number]).columns
        print(f"   Numeric columns: {list(numeric_columns)}")
        
        for col in numeric_columns:
            if col in features_processed.columns:
                # Заполняем пропуски медианой
                if features_processed[col].isnull().any():
                    median_val = features_processed[col].median()
                    features_processed[col] = features_processed[col].fillna(median_val)
                    print(f"   Filled NaN in {col} with median: {median_val}")
        
        # 4. Проверяем финальные данные
        print(f"   Final features shape: {features_processed.shape}")
        print(f"   Final dtypes: {features_processed.dtypes.unique()}")
        
        # 5. Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features_processed, targets, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=targets
        )
        
        print(f"📈 Train set: {X_train.shape}, positives: {y_train.mean():.3f}")
        print(f"📊 Test set: {X_test.shape}, positives: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test

    def initialize_models(self):
        """Инициализация моделей для экспериментов"""
        print("🤖 Initializing models...")
        
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1,
                max_depth=10, min_samples_split=5
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1,
                max_depth=8, learning_rate=0.05, subsample=0.8
            )
        }
        
        return self.models

    def train_models(self, X_train, y_train):
        """Обучение всех моделей"""
        print("🚀 Training models...")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"🔧 Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.results[name] = model
                print(f"✅ {name} trained successfully")
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                # Пропускаем проблемную модель
                continue
        
        return self.results

    def evaluate_models(self, X_test, y_test):
        """Оценка качества моделей"""
        print("📊 Evaluating models...")
        
        evaluation_results = {}
        
        for name, model in self.results.items():
            print(f"\n📈 Evaluating {name}:")
            
            try:
                # Предсказания
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Метрики
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                evaluation_results[name] = {
                    'roc_auc': roc_auc,
                    'precision': report['1']['precision'] if '1' in report else 0,
                    'recall': report['1']['recall'] if '1' in report else 0,
                    'f1_score': report['1']['f1-score'] if '1' in report else 0,
                    'accuracy': report['accuracy']
                }
                
                print(f"   ROC-AUC: {roc_auc:.4f}")
                if '1' in report:
                    print(f"   Precision: {report['1']['precision']:.4f}")
                    print(f"   Recall: {report['1']['recall']:.4f}")
                    print(f"   F1-Score: {report['1']['f1-score']:.4f}")
                print(f"   Accuracy: {report['accuracy']:.4f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {name}: {e}")
                evaluation_results[name] = {
                    'roc_auc': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0,
                    'accuracy': 0
                }
        
        return evaluation_results

    def plot_feature_importance(self, model, feature_names, top_n=20):
        """Визуализация важности признаков"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                print("⚠️ Model doesn't support feature importance")
                return None, None
            
            # Сортируем признаки по важности
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance")
            plt.bar(range(min(top_n, len(importance))), importance[indices][:top_n])
            plt.xticks(range(min(top_n, len(importance))), 
                      [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
            plt.tight_layout()
            
            # Сохраняем график вместо показа
            plots_dir = os.path.join(root_dir, 'results', 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
            plt.close()
            
            print("✅ Feature importance plot saved")
            return indices, importance
            
        except Exception as e:
            print(f"❌ Error creating feature importance plot: {e}")
            return None, None

    def save_models(self, models_dir='models'):
        """Сохранение обученных моделей"""
        print("💾 Saving models...")
        
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.results.items():
            try:
                model_path = os.path.join(models_dir, f'{name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'label_encoders': self.label_encoders
                    }, f)
                print(f"✅ {name} saved to {model_path}")
            except Exception as e:
                print(f"❌ Error saving {name}: {e}")

    def save_results(self, results, results_dir='results'):
        """Сохранение результатов экспериментов"""
        print("💾 Saving results...")
        
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # Сохраняем метрики в CSV
            results_df = pd.DataFrame(results).T
            results_path = os.path.join(results_dir, 'model_results.csv')
            results_df.to_csv(results_path)
            print(f"✅ Results saved to {results_path}")
            
            # Сохраняем детальный отчет
            report_path = os.path.join(results_dir, 'detailed_report.txt')
            with open(report_path, 'w') as f:
                f.write("MODEL TRAINING RESULTS\n")
                f.write("=" * 50 + "\n\n")
                for name, metrics in results.items():
                    f.write(f"{name.upper()}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
            
            return results_df
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None

def main():
    """Главная функция обучения моделей"""
    print("=" * 60)
    print("🚀 STAGE 4: MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    try:
        # Инициализация
        trainer = ModelTrainer()
        
        # 1. Загрузка данных
        print("\n1. 📥 LOADING DATA")
        print("-" * 40)
        
        # Используем абсолютные пути
        features_path = os.path.join(root_dir, 'ozon', 'data', 'processed', 'features.parquet')
        targets_path = os.path.join(root_dir, 'ozon', 'data', 'processed', 'targets.parquet')
        
        print(f"Looking for features at: {features_path}")
        print(f"Looking for targets at: {targets_path}")
        
        if not os.path.exists(features_path) or not os.path.exists(targets_path):
            print("❌ Processed data not found! Run feature engineering first.")
            return None, None
        
        features, targets = trainer.load_data(features_path, targets_path)
        
        # 2. Подготовка данных
        print("\n2. 🛠️ PREPARING DATA")
        print("-" * 40)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            features, targets, 
            test_size=0.2, 
            random_state=42
        )
        
        # Проверяем что данные валидны
        if X_train.shape[1] == 0:
            print("❌ No features left after preprocessing!")
            return None, None
        
        # 3. Инициализация моделей
        print("\n3. 🤖 INITIALIZING MODELS")
        print("-" * 40)
        
        models = trainer.initialize_models()
        
        # 4. Обучение моделей
        print("\n4. 🚀 TRAINING MODELS")
        print("-" * 40)
        
        trained_models = trainer.train_models(X_train, y_train)
        
        if not trained_models:
            print("❌ All models failed to train!")
            return None, None
        
        # 5. Оценка моделей
        print("\n5. 📊 EVALUATING MODELS")
        print("-" * 40)
        
        results = trainer.evaluate_models(X_test, y_test)
        
        # 6. Анализ лучшей модели
        print("\n6. 🏆 BEST MODEL ANALYSIS")
        print("-" * 40)
        
        # Находим лучшую модель по ROC-AUC
        if results:
            best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
            best_model = trained_models[best_model_name]
            best_result = results[best_model_name]
            
            print(f"🎉 Best model: {best_model_name}")
            print(f"📊 ROC-AUC: {best_result['roc_auc']:.4f}")
            print(f"🎯 Precision: {best_result['precision']:.4f}")
            print(f"🔍 Recall: {best_result['recall']:.4f}")
            print(f"⚡ F1-Score: {best_result['f1_score']:.4f}")
            
            # Визуализация важности признаков для tree-based моделей
            if best_model_name in ['random_forest', 'lightgbm']:
                print("\n📊 Feature Importance Analysis:")
                trainer.plot_feature_importance(best_model, X_train.columns.tolist())
        else:
            print("❌ No models evaluated successfully!")
        
        # 7. Сохранение результатов
        print("\n7. 💾 SAVING RESULTS")
        print("-" * 40)
        
        models_dir = os.path.join(root_dir, 'models')
        results_dir = os.path.join(root_dir, 'results')
        
        trainer.save_models(models_dir)
        results_df = trainer.save_results(results, results_dir)
        
        print("\n" + "=" * 60)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return trained_models, results_df
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()