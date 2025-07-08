import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignorar avisos para uma saída mais limpa
warnings.filterwarnings('ignore')

# --- 1. CONFIGURAÇÕES E CONSTANTES ---
DATA_PATH = 'data/'
TARGET = 'INADIMPLENTE'
ID_COLS = ['ID_CLIENTE', 'SAFRA_REF']
SUBMISSION_FILE = 'submissao_case.csv'

# --- 2. FUNÇÕES DE PRÉ-PROCESSAMENTO E ENGENHARIA DE ATRIBUTOS ---

def load_data(path: str) -> dict:
    """Carrega todos os arquivos CSV do diretório especificado, usando ';' como separador."""
    print("Iniciando carregamento dos dados (usando ';' como separador)...")
    try:
        dfs = {
            'cadastral': pd.read_csv(f'{path}base_cadastral.csv', sep=';'),
            'info': pd.read_csv(f'{path}base_info.csv', sep=';'),
            'dev': pd.read_csv(f'{path}base_pagamentos_desenvolvimento.csv', sep=';'),
            'teste': pd.read_csv(f'{path}base_pagamentos_teste.csv', sep=';')
        }
        print("Dados carregados com sucesso.")
        
        # Verificação rápida para garantir que a leitura funcionou
        print("Colunas da base de desenvolvimento após correção:", dfs['dev'].columns.to_list())
        
        return dfs
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique o caminho: {e.path}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao ler os arquivos: {e}")
        print("Verifique se todos os arquivos estão realmente separados por ';'.")
        return None

def create_target_variable(df_dev: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo 'INADIMPLENTE' na base de desenvolvimento."""
    print("Criando variável alvo...")
    # Converte colunas de data para datetime
    df_dev['DATA_VENCIMENTO'] = pd.to_datetime(df_dev['DATA_VENCIMENTO'])
    df_dev['DATA_PAGAMENTO'] = pd.to_datetime(df_dev['DATA_PAGAMENTO'])

    # Calcula dias de atraso
    df_dev['DIAS_ATRASO'] = (df_dev['DATA_PAGAMENTO'] - df_dev['DATA_VENCIMENTO']).dt.days

    # Define a flag de inadimplência (atraso >= 5 dias)
    df_dev[TARGET] = (df_dev['DIAS_ATRASO'] >= 5).astype(int)
    
    print(f"Distribuição da variável alvo:\n{df_dev[TARGET].value_counts(normalize=True)}")
    return df_dev

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica engenharia de atributos no DataFrame combinado."""
    print(f"Iniciando engenharia de atributos para dataframe com {df.shape[0]} linhas...")
    
    # Garantir que as colunas de data estejam no formato correto
    for col in ['DATA_CADASTRO', 'DATA_EMISSAO_DOCUMENTO', 'DATA_VENCIMENTO']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # 1. Features baseadas em datas
    df['PRAZO_COBRANCA'] = (df['DATA_VENCIMENTO'] - df['DATA_EMISSAO_DOCUMENTO']).dt.days
    df['DIAS_DESDE_CADASTRO'] = (df['DATA_VENCIMENTO'] - df['DATA_CADASTRO']).dt.days
    df['MES_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.month
    df['DIA_SEMANA_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.dayofweek

    # 2. Features categóricas e textuais
    # Simplifica domínios de email para reduzir cardinalidade
    common_domains = ['gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com', 'live.com']
    df['TIPO_DOMINIO_EMAIL'] = df['DOMINIO_EMAIL'].apply(
        lambda x: 'comum' if x in common_domains else ('corporativo' if pd.notna(x) else 'outro')
    )

    # 3. Features de interação
    # Evita divisão por zero
    df['RENDA_POR_FUNCIONARIO'] = df['RENDA_MES_ANTERIOR'] / (df['NO_FUNCIONARIOS'] + 1)
    df['VALOR_PAGAR_SOBRE_RENDA'] = df['VALOR_A_PAGAR'] / (df['RENDA_MES_ANTERIOR'] + 1)

    # 4. Features de comportamento histórico (usando groupby e shift)
    df = df.sort_values(by=ID_COLS)
    grouped = df.groupby('ID_CLIENTE')
    
    df['VALOR_PAGAR_MES_ANT'] = grouped['VALOR_A_PAGAR'].shift(1)
    df['TAXA_MES_ANT'] = grouped['TAXA'].shift(1)
    
    # Média móvel do valor a pagar nos últimos 3 meses
    df['VALOR_PAGAR_MEDIA_3M'] = grouped['VALOR_A_PAGAR'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # Se a base tiver DIAS_ATRASO (apenas para o treino), cria features históricas de inadimplência
    if 'DIAS_ATRASO' in df.columns:
        df['DIAS_ATRASO_MES_ANT'] = grouped['DIAS_ATRASO'].shift(1)
        df['INADIMPLENTE_MES_ANT'] = grouped[TARGET].shift(1)
        df['INADIMPLENCIA_ACUMULADA'] = grouped[TARGET].transform(
            lambda x: x.shift(1).expanding().sum()
        )
        df['PAGAMENTOS_TOTAIS'] = grouped[TARGET].transform(
            lambda x: x.shift(1).expanding().count()
        )
        df['TAXA_INADIMPLENCIA_CLIENTE'] = df['INADIMPLENCIA_ACUMULADA'] / (df['PAGAMENTOS_TOTAIS'] + 1)

    print("Engenharia de atributos concluída.")
    return df

def preprocess_final(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    """Prepara o dataframe final para o modelo."""
    print(f"Pré-processamento final. Is_train={is_train}")
    
    # Selecionar colunas a serem removidas
    cols_to_drop = [
        'DATA_CADASTRO', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO',
        'DIAS_ATRASO', 'DOMINIO_EMAIL', 'INADIMPLENCIA_ACUMULADA', 'PAGAMENTOS_TOTAIS'
    ]
    # Remove o target apenas se não for treino
    if not is_train:
        cols_to_drop.append(TARGET)

    df_processed = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Lidar com variáveis categóricas via One-Hot Encoding
    # ---- INÍCIO DA CORREÇÃO ----
    # Identifica as colunas categóricas
    categorical_features = df_processed.select_dtypes(include=['object', 'category']).columns
    # Remove as colunas de ID da lista de features a serem transformadas
    categorical_features = [col for col in categorical_features if col not in ID_COLS]
    # ---- FIM DA CORREÇÃO ----

    df_processed = pd.get_dummies(df_processed, columns=categorical_features, dummy_na=True)
    
    # Tratamento simples de nulos para colunas numéricas restantes
    for col in df_processed.select_dtypes(include=np.number).columns:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def plot_feature_importance(model, features):
    """Plota a importância das features do modelo LightGBM."""
    plt.figure(figsize=(10, 12))
    lgb.plot_importance(model, max_num_features=30, height=0.8)
    plt.title("Importância das Features (Top 30)")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Gráfico de importância das features salvo como 'feature_importance.png'.")

# --- 3. FLUXO PRINCIPAL DE EXECUÇÃO ---

def main():
    """Função principal que orquestra todo o pipeline."""
    
    # 1. Carregar os dados
    data_dict = load_data(DATA_PATH)
    if not data_dict:
        return

    # 2. Criar a variável alvo na base de desenvolvimento
    df_dev = create_target_variable(data_dict['dev'])

    # 3. Combinar bases cadastrais e de informações mensais
    df_dev_full = pd.merge(df_dev, data_dict['cadastral'], on='ID_CLIENTE', how='left')
    df_dev_full = pd.merge(df_dev_full, data_dict['info'], on=ID_COLS, how='left')
    
    df_teste_full = pd.merge(data_dict['teste'], data_dict['cadastral'], on='ID_CLIENTE', how='left')
    df_teste_full = pd.merge(df_teste_full, data_dict['info'], on=ID_COLS, how='left')
    
    # 4. Engenharia de Atributos
    df_dev_featured = feature_engineering(df_dev_full.copy())
    
    # Para o teste, as features históricas devem ser criadas a partir do histórico de dev
    # Uma forma é concatenar, criar features, e depois separar.
    df_teste_full[TARGET] = -1 # Placeholder para concatenar
    combined_df = pd.concat([df_dev_full, df_teste_full], ignore_index=True)
    combined_featured = feature_engineering(combined_df.copy())
    
    df_dev_featured = combined_featured[combined_featured[TARGET] != -1].copy()
    df_teste_featured = combined_featured[combined_featured[TARGET] == -1].copy()

    # 5. Pré-processamento final (continuação)
    df_train_processed = preprocess_final(df_dev_featured, is_train=True)
    df_test_processed = preprocess_final(df_teste_featured, is_train=False)

    # --- INÍCIO DA CORREÇÃO ---
    # 6. Alinhamento Robusto de Colunas e Preparação dos Dados
    print("Alinhando colunas entre treino e teste de forma robusta...")

    # Pega todas as colunas de treino, exceto IDs e Target
    train_feature_cols = df_train_processed.drop(columns=ID_COLS + [TARGET]).columns
    # Pega todas as colunas de teste, exceto IDs
    test_feature_cols = df_test_processed.drop(columns=ID_COLS).columns

    # Cria uma lista mestra com a união de todas as features de treino e teste
    all_features = sorted(list(set(train_feature_cols) | set(test_feature_cols)))
    print(f"Total de features únicas encontradas em ambos os conjuntos: {len(all_features)}")

    # Prepara o DataFrame final de treino (X) e o target (y)
    # Reindexa o DataFrame de treino para garantir que ele tenha todas as features.
    # Colunas que não existiam antes serão preenchidas com 0.
    X = df_train_processed.reindex(columns=all_features, fill_value=0)
    y = df_train_processed[TARGET]

    # Prepara o DataFrame final de teste (X_teste)
    # Reindexa da mesma forma para garantir a correspondência exata.
    X_teste = df_test_processed.reindex(columns=all_features, fill_value=0)

    # Garante que a ordem das colunas é a mesma
    X_teste = X_teste[X.columns]
    # --- FIM DA CORREÇÃO ---

    # 7. Divisão para validação (temporal)
    # Usaremos as últimas safras da base de desenvolvimento para validação
    # A lógica de split usa o dataframe original ANTES do reindex para pegar a safra
    safras_ordenadas = sorted(df_train_processed['SAFRA_REF'].unique())
    safra_corte = safras_ordenadas[-3] # Usa as 2 últimas safras para validar

    idx_treino = df_train_processed['SAFRA_REF'] < safra_corte
    idx_valid = df_train_processed['SAFRA_REF'] >= safra_corte

    # Usa o índice para fatiar os dataframes já alinhados (X, y)
    X_train, y_train = X[idx_treino], y[idx_treino]
    X_valid, y_valid = X[idx_valid], y[idx_valid]
    
    print(f"Tamanho do conjunto de treino: {X_train.shape}")
    print(f"Tamanho do conjunto de validação: {X_valid.shape}")
    print(f"Tamanho do conjunto de teste final: {X_teste.shape}")

    # 8. Treinamento do Modelo
    print("\nIniciando treinamento do modelo LightGBM...")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
    }

    model = lgb.LGBMClassifier(**lgb_params)

    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # 9. Avaliação
    valid_preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_preds)
    print(f"AUC na validação temporal: {auc:.4f}")

    # 10. Treinamento final com todos os dados de desenvolvimento e Geração de Submissão
    print("Treinando modelo final com todos os dados de desenvolvimento...")
    model.fit(X, y) # Retreina com todos os dados
    
    print("Gerando previsões para a base de teste...")
    test_preds = model.predict_proba(X_teste)[:, 1]

    df_submission = pd.DataFrame({
        'ID_CLIENTE': df_teste_featured['ID_CLIENTE'],
        'SAFRA_REF': df_teste_featured['SAFRA_REF'],
        'PROBABILIDADE_INADIMPLENCIA': test_preds
    })

    df_submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Arquivo de submissão '{SUBMISSION_FILE}' criado com sucesso!")
    
    # 11. Plotar importância das features
    plot_feature_importance(model, X.columns)


if __name__ == '__main__':
    main()