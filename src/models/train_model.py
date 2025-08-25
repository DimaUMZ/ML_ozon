#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
        
    def load_data(self, features_path, targets_path):
        """Загрузка подготовленных данных"""
        print("📥 Loading processed data...")
        
        features = pd.read_parquet(features_path)
        targets = pd.read_parquet(targets_path)
        
        print(f"📊 Features shape: {features.shape}")
        print(f"🎯 Targets shape: {targets.shape}")
        print(f"📋 Features columns: {list(features.columns)}")
        
        return features, targets['target']  # Извлекаем Series из DataFrame

    def prepare_data(self, features, targets, test_size=0.2, random_state=42):
        """Подготовка данных для обучения"""
        print("🛠️ Preparing data for training...")
        
        # Создаем копию чтобы не изменять оригинальные данные
        features_processed = features.copy()
        
        # Преобразуем временные колонки в числовые
        timestamp_columns = ['timestamp', 'created_timestamp', 'last_status_timestamp', 'date', 'created_date']
        
        for col in timestamp_columns:
            if col in features_processed.columns:
                print(f"   Converting {col} to numeric...")
                # Преобразуем в Unix timestamp (количество секунд)
                features_processed[col] = pd.to_datetime(features_processed[col]).astype('int64') // 10**9
        
        # Удаляем нечисловые колонки которые не нужны для обучения
        columns_to_drop = ['action_type', 'action_widget', 'last_status', 'itemname']
        for col in columns_to_drop:
            if col in features_processed.columns:
                print(f"   Dropping column: {col}")
                features_processed = features_processed.drop(columns=[col])
        
        # Проверяем что все колонки числовые
        print(f"   Final columns: {list(features_processed.columns)}")
        print(f"   Column dtypes: {features_processed.dtypes.unique()}")
        
        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features_processed, targets, test_size=test_size, random_state=random_state, stratify=targets
        )
        
        print(f"📈 Train set: {X_train.shape}, {y_train.shape}")
        print(f"📊 Test set: {X_test.shape}, {y_test.shape}")
        print(f"🎯 Class balance - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test

    def initialize_models(self):
        """Инициализация моделей для экспериментов"""
        print("🤖 Initializing models...")
        
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
            )
        }
        
        return self.models

    def train_models(self, X_train, y_train):
        """Обучение всех моделей"""
        print("🚀 Training models...")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"🔧 Training {name}...")
            model.fit(X_train, y_train)
            self.results[name] = model
            print(f"✅ {name} trained successfully")
        
        return self.results

    def evaluate_models(self, X_test, y_test):
        """Оценка качества моделей"""
        print("📊 Evaluating models...")
        
        evaluation_results = {}
        
        for name, model in self.results.items():
            print(f"\n📈 Evaluating {name}:")
            
            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Метрики
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            evaluation_results[name] = {
                'roc_auc': roc_auc,
                'precision': report['1']['precision'], # type: ignore
                'recall': report['1']['recall'], # type: ignore
                'f1_score': report['1']['f1-score'], # type: ignore
                'accuracy': report['accuracy'] # type: ignore
            }
            
            print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   Precision: {report['1']['precision']:.4f}") # type: ignore
            print(f"   Recall: {report['1']['recall']:.4f}") # type: ignore
            print(f"   F1-Score: {report['1']['f1-score']:.4f}") # type: ignore
            print(f"   Accuracy: {report['accuracy']:.4f}") # type: ignore
        
        return evaluation_results

    def cross_validate(self, X, y, cv=3):
        """Кросс-валидация моделей"""
        print("🔍 Cross-validating models...")
        
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"📊 Cross-validating {name}...")
            
            # ROC-AUC кросс-валидация
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            
            cv_results[name] = {
                'mean_roc_auc': cv_scores.mean(),
                'std_roc_auc': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"   Mean ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return cv_results

    def plot_feature_importance(self, model, feature_names, top_n=20):
        """Визуализация важности признаков"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            print("⚠️ Model doesn't support feature importance")
            return
        
        # Сортируем признаки по важности
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance")
        plt.bar(range(min(top_n, len(importance))), importance[indices][:top_n])
        plt.xticks(range(min(top_n, len(importance))), 
                  [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return indices, importance

    def save_models(self, models_dir='models'):
        """Сохранение обученных моделей"""
        print("💾 Saving models...")
        
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.results.items():
            model_path = os.path.join(models_dir, f'{name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ {name} saved to {model_path}")

    def save_results(self, results, results_dir='results'):
        """Сохранение результатов экспериментов"""
        print("💾 Saving results...")
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Сохраняем метрики в CSV
        results_df = pd.DataFrame(results).T
        results_path = os.path.join(results_dir, 'model_results.csv')
        results_df.to_csv(results_path)
        print(f"✅ Results saved to {results_path}")
        
        return results_df

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
        
        features_path = os.path.expanduser('~/ozon/data/processed/features.parquet')
        targets_path = os.path.expanduser('~/ozon/data/processed/targets.parquet')
        
        features, targets = trainer.load_data(features_path, targets_path)
        
        # 2. Подготовка данных
        print("\n2. 🛠️ PREPARING DATA")
        print("-" * 40)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
        features, targets, 
        test_size=0.2, 
        random_state=42
        )
        # X_train, X_test, y_train, y_test = trainer.prepare_data(features, targets)
        
        # 3. Инициализация моделей
        print("\n3. 🤖 INITIALIZING MODELS")
        print("-" * 40)
        
        models = trainer.initialize_models()
        
        # 4. Обучение моделей
        print("\n4. 🚀 TRAINING MODELS")
        print("-" * 40)
        
        trained_models = trainer.train_models(X_train, y_train)
        
        # 5. Оценка моделей
        print("\n5. 📊 EVALUATING MODELS")
        print("-" * 40)
        
        results = trainer.evaluate_models(X_test, y_test)
        
        # 6. Кросс-валидация
        # print("\n6. 🔍 CROSS-VALIDATION")
        # print("-" * 40)
        
        # cv_results = trainer.cross_validate(features, targets, cv=3)
        
        # 7. Анализ лучшей модели
        print("\n7. 🏆 BEST MODEL ANALYSIS")
        print("-" * 40)
        
        # Находим лучшую модель по ROC-AUC
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
            trainer.plot_feature_importance(best_model, features.columns.tolist())
        
        # 8. Сохранение результатов
        print("\n8. 💾 SAVING RESULTS")
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