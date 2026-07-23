"""
check_columns.py  ← Ejecutar PRIMERO para verificar nombres de columnas
Ejecutar: python check_columns.py

Imprime los nombres exactos de columna del Excel y valores únicos
de columnas clave, para verificar que data_loader.py los mapea correctamente.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Buscar el Excel
paths = [
    PROJECT_ROOT / "data" / "raw" / "Base_Migracion_2009-2026jun.xlsx",
    PROJECT_ROOT / "Laboratorio 1. Series de Tiempo 2026 - Base_Migracion_2009-2026jun.xlsx",
]

xlsx_path = None
for p in paths:
    if p.exists():
        xlsx_path = p
        break

if xlsx_path is None:
    print("ERROR: No se encontró el Excel. Verificar rutas en paths[]")
    sys.exit(1)

print(f"Leyendo: {xlsx_path.name}\n")
df = pd.read_excel(xlsx_path, nrows=3)

print("=" * 50)
print("COLUMNAS (nombres exactos en el Excel):")
print("=" * 50)
for i, col in enumerate(df.columns):
    print(f"  [{i}] '{col}'")

print("\n" + "=" * 50)
print("PRIMERAS 3 FILAS:")
print("=" * 50)
print(df.to_string())

print("\n" + "=" * 50)
print("VALORES ÚNICOS POR COLUMNA (muestra):")
print("=" * 50)
df_full = pd.read_excel(xlsx_path)
print(f"\nShape: {df_full.shape}")
for col in df_full.columns:
    uniques = df_full[col].dropna().unique()
    n = len(uniques)
    sample = uniques[:5].tolist()
    print(f"\n  '{col}' ({n} únicos): {sample}")
