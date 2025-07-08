import pandas as pd

df_sub = pd.read_csv('submissao_case.csv')
df_teste_original = pd.read_csv('data/base_pagamentos_teste.csv', sep=';')

print("--- Verificação do Arquivo de Submissão ---")

# 1. Checagem de Colunas
colunas_esperadas = ['ID_CLIENTE', 'SAFRA_REF', 'PROBABILIDADE_INADIMPLENCIA']
if list(df_sub.columns) == colunas_esperadas:
    print("✅ Colunas: OK")
else:
    print(f"❌ Colunas: ERRO! Esperado: {colunas_esperadas}, Encontrado: {list(df_sub.columns)}")

# 2. Checagem do Número de Linhas
if len(df_sub) == len(df_teste_original):
    print(f"✅ Número de Linhas: OK ({len(df_sub)})")
else:
    print(f"❌ Número de Linhas: ERRO! Esperado: {len(df_teste_original)}, Encontrado: {len(df_sub)}")

# 3. Checagem do Range das Probabilidades
probs = df_sub['PROBABILIDADE_INADIMPLENCIA']
if probs.min() >= 0 and probs.max() <= 1:
    print(f"✅ Range das Probabilidades: OK (Min: {probs.min():.4f}, Max: {probs.max():.4f})")
else:
    print("❌ Range das Probabilidades: ERRO! Existem valores fora do intervalo [0, 1].")

# 4. Checagem de valores nulos
if df_sub.isnull().sum().sum() == 0:
    print("✅ Valores Nulos: OK (Nenhum valor nulo encontrado)")
else:
    print("❌ Valores Nulos: ERRO! Existem valores nulos no arquivo.")
