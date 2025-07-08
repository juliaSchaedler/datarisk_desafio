import pandas as pd
from config import ID_COLS, TARGET

def create_target_variable(df_dev: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo 'INADIMPLENTE' na base de desenvolvimento."""
    df_dev['DATA_VENCIMENTO'] = pd.to_datetime(df_dev['DATA_VENCIMENTO'])
    df_dev['DATA_PAGAMENTO'] = pd.to_datetime(df_dev['DATA_PAGAMENTO'])
    df_dev['DIAS_ATRASO'] = (df_dev['DATA_PAGAMENTO'] - df_dev['DATA_VENCIMENTO']).dt.days
    df_dev[TARGET] = (df_dev['DIAS_ATRASO'] >= 5).astype(int)
    return df_dev

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica engenharia de atributos avançada."""
    print(f"Iniciando engenharia de atributos para dataframe com {df.shape[0]} linhas...")
    df = df.sort_values(by=ID_COLS)
    
    # Conversão de Datas
    for col in ['DATA_CADASTRO', 'DATA_EMISSAO_DOCUMENTO', 'DATA_VENCIMENTO']:
        if col in df.columns: df[col] = pd.to_datetime(df[col])

    # Features de Data
    df['PRAZO_COBRANCA'] = (df['DATA_VENCIMENTO'] - df['DATA_EMISSAO_DOCUMENTO']).dt.days
    df['DIAS_DESDE_CADASTRO'] = (df['DATA_VENCIMENTO'] - df['DATA_CADASTRO']).dt.days
    df['MES_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.month
    
    # Features Categóricas
    common_domains = ['gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com', 'live.com']
    df['TIPO_DOMINIO_EMAIL'] = df['DOMINIO_EMAIL'].apply(lambda x: 'comum' if x in common_domains else 'corporativo' if pd.notna(x) else 'outro')

    # Features de Comportamento Histórico (agrupado por cliente)
    grouped = df.groupby('ID_CLIENTE')
    df['VALOR_PAGAR_MEDIA_3M'] = grouped['VALOR_A_PAGAR'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['TAXA_MEDIA_3M'] = grouped['TAXA'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    # Feature de Tendência: Compara o valor atual com a média histórica do cliente.
    df['VALOR_PAGAR_VS_MEDIA_3M'] = df['VALOR_A_PAGAR'] / (df['VALOR_PAGAR_MEDIA_3M'] + 1e-6)

    # Feature de Interação: Renda por funcionário.
    df['RENDA_POR_FUNCIONARIO'] = df['RENDA_MES_ANTERIOR'] / (df['NO_FUNCIONARIOS'] + 1)
    
    if 'DIAS_ATRASO' in df.columns:
        df['INADIMPLENTE_MES_ANT'] = grouped[TARGET].shift(1)
        df['TAXA_INADIMPLENCIA_CLIENTE'] = grouped[TARGET].transform(lambda x: x.shift(1).expanding().mean())

    # Remover colunas originais que não serão mais usadas diretamente
    df = df.drop(columns=['DOMINIO_EMAIL'])
    print("Engenharia de atributos concluída.")
    return df
