# project_scoping.py
from datetime import datetime
import json

class RecommendationSystemScope:
    """
    Документ определения scope проекта рекомендательной системы
    На основе EDA анализа от 2025-08-21
    """
    
    def __init__(self, eda_results):
        self.eda = eda_results
        self.scope_date = datetime.now().strftime("%Y-%m-%d")
        
    def generate_scoping_document(self):
        """Генерация полного документа scope"""
        return {
            "project_name": "E-commerce Recommendation System",
            "version": "1.0",
            "scoping_date": self.scope_date,
            "business_objective": self._get_business_objective(),
            "problem_type": "Learning to Rank (LTR) для рекомендаций",
            "success_metrics": self._get_success_metrics(),
            "data_sources": self._get_data_sources(),
            "scope_inclusions": self._get_scope_inclusions(),
            "scope_exclusions": self._get_scope_exclusions(),
            "timeline": self._get_timeline(),
            "risks_and_assumptions": self._get_risks()
        }
    
    def _get_business_objective(self):
        return {
            "primary_goal": "Увеличение конверсии и среднего чека через персонализированные рекомендации",
            "secondary_goals": [
                "Уменьшение bounce rate",
                "Увеличение времени на сайте",
                "Снижение процента отмененных заказов"
            ]
        }
    
    def _get_success_metrics(self):
        return {
            "business_metrics": [
                "CTR (Click-Through Rate) рекомендаций",
                "Conversion Rate в покупки",
                "Увеличение среднего количества товаров в заказе",
                "Снижение процента отмен заказов на 15%"
            ],
            "technical_metrics": [
                "Precision@K (K=5, 10)",
                "Recall@K (K=5, 10)",
                "NDCG@K (K=5, 10)",
                "MAP (Mean Average Precision)",
                "AUC-ROC для бинарной классификации"
            ]
        }
    
    def _get_data_sources(self):
        return {
            "primary_data": [
                {"name": "User Interactions", "rows": self.eda["test_analysis"]["rows"], "purpose": "User behavior features"},
                {"name": "Items Catalog", "rows": self.eda["items_analysis"]["rows"], "purpose": "Item metadata and features"},
                {"name": "Orders", "rows": self.eda["orders_analysis"]["rows"], "purpose": "Purchase signals and negative sampling"},
                {"name": "Event Tracker", "rows": self.eda["tracker_analysis"]["rows"], "purpose": "Detailed user actions"}
            ],
            "coverage_analysis": {
                "user_coverage": f"{(self.eda['orders_analysis']['unique_users'] / self.eda['test_analysis']['unique_users'] * 100):.1f}%",
                "item_coverage": f"{(self.eda['orders_analysis']['unique_items'] / self.eda['items_analysis']['rows'] * 100):.1f}%"
            }
        }
    
    def _get_scope_inclusions(self):
        return [
            "Рекомендации на главной странице",
            "Рекомендации в карточке товара ('Похожие товары')",
            "Рекомендации в корзине",
            "Персонализированные email-рассылки",
            "Оффлайн-рекомендации для CRM"
        ]
    
    def _get_scope_exclusions(self):
        return [
            "Реал-тайм рекомендации (первая версия)",
            "Рекомендации для новых пользователей (cold start)",
            "Мультимодальные embeddings (изображения, текст)",
            "A/B тестирование нескольких моделей одновременно"
        ]
    
    def _get_timeline(self):
        return {
            "phase_1": {"duration": "2 недели", "tasks": ["Feature Engineering", "Baseline модель"]},
            "phase_2": {"duration": "3 недели", "tasks": ["Оптимизация модели", "Оффлайн-валидация"]},
            "phase_3": {"duration": "1 неделя", "tasks": ["Пилотное A/B тестирование"]},
            "phase_4": {"duration": "2 недели", "tasks": ["Продакшн деплой", "Мониторинг"]}
        }
    
    def _get_risks(self):
        return {
            "data_risks": [
                "Высокий процент отмененных заказов (37.6%)",
                "Ограниченный охват товаров в трекере (всего 10 unique_items)",
                "Синтетический характер данных (даты в 2025 году)"
            ],
            "technical_risks": [
                "Cold start problem для новых пользователей/товаров",
                "Классовый дисбаланс (мало покупок относительно просмотров)",
                "Временной дрейф пользовательских предпочтений"
            ],
            "mitigation_strategies": [
                "Использование контентных фич для cold start",
                "Правильное взвешивание и сэмплирование",
                "Регулярное переобучение модели"
            ]
        }

# Использование
if __name__ == "__main__":
    # Загрузка EDA результатов
    path = '/home/dima/ozon/EDA (Exploratory Data Analysis)/eda_results.json'
    with open(path, 'r') as f:
        eda_data = json.load(f)
    
    # Генерация scoping документа
    scoper = RecommendationSystemScope(eda_data)
    scope_doc = scoper.generate_scoping_document()
    
    # Сохранение документа
    with open('/home/dima/ozon/scripts/project_scoping.json', 'w', encoding='utf-8') as f:
        json.dump(scope_doc, f, ensure_ascii=False, indent=2)
    
    print("Project Scoping Document создан успешно!")
