#!/usr/bin/env python3
"""
Скрипт для обучения моделей
"""

import sys
import os

# Добавляем src в путь
sys.path.append('src')

from models.train_model import main

if __name__ == "__main__":
    main()