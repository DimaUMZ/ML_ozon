#!/usr/bin/env python3
"""
Тестирование окружения с исправлением проблемы AVX
"""

import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

# Отключаем проверку CPU features перед импортом polars
import os
os.environ['POLARS_SKIP_CPU_CHECK'] = '1'

# Проверяем базовые библиотеки
try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print("✗ NumPy import failed:", e)

try:
    import pandas as pd
    print("✓ Pandas imported successfully") 
except ImportError as e:
    print("✗ Pandas import failed:", e)

try:
    # Теперь пробуем импортировать polars с отключенной проверкой
    import polars as pl
    print("✓ Polars imported successfully with CPU check disabled")
    
    # Проверяем базовые операции
    test_df = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print("✓ Basic Polars operations work")
    
except ImportError as e:
    print("✗ Polars import failed:", e)
except Exception as e:
    print("✗ Polars operation failed:", e)

# Проверяем память
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / 1024**3:.1f} GB")
    print(f"Available RAM: {memory.available / 1024**3:.1f} GB")
except ImportError as e:
    print("✗ psutil import failed:", e)

print("Environment test completed!")