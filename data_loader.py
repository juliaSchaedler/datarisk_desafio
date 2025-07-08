import pandas as pd
from config import DATA_PATH

def load_data() -> dict:
    """Carrega todos os arquivos CSV, usando ';' como separador."""
    print("Iniciando carregamento dos dados...")
    try:
        dfs = {
            'cadastral': pd.read_csv(f'{DATA_PATH}base_cadastral.csv', sep=';'),
            'info': pd.read_csv(f'{DATA_PATH}base_info.csv', sep=';'),
            'dev': pd.read_csv(f'{DATA_PATH}base_pagamentos_desenvolvimento.csv', sep=';'),
            'teste': pd.read_csv(f'{DATA_PATH}base_pagamentos_teste.csv', sep=';')
        }
        print("Dados carregados com sucesso.")
        return dfs
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique o caminho: {e.path}")
        exit()
