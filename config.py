"""
Arquivo de configuração central para o projeto.
"""

# Caminhos
DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'
SUBMISSION_FILE = 'submissao_case.csv'

# Constantes do Modelo
ID_COLS = ['ID_CLIENTE', 'SAFRA_REF']
TARGET = 'INADIMPLENTE'

# Parâmetros de Otimização e Modelo
#OPTUNA_TRIALS = 25  # Número de tentativas para o Optuna. Aumente para uma busca melhor.
LGBM_FIXED_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 2000,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
}
