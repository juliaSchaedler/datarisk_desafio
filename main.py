import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from config import ID_COLS, TARGET, SUBMISSION_FILE, OUTPUT_PATH, LGBM_FIXED_PARAMS
from data_loader import load_data
from feature_engineering import create_target_variable, feature_engineering
from model_trainer import train_final_model 
from evaluation import evaluate_model


def preprocess_for_model(df: pd.DataFrame, is_train=True):
    """Aplica o pré-processamento final para o modelo."""
    cols_to_drop = [c for c in ['DATA_CADASTRO', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO', 'DIAS_ATRASO'] if c in df.columns]
    df_processed = df.drop(columns=cols_to_drop)

    categorical_features = df_processed.select_dtypes(include=['object', 'category']).columns
    categorical_features = [col for col in categorical_features if col not in ID_COLS]
    df_processed = pd.get_dummies(df_processed, columns=categorical_features, dummy_na=True, dtype=float)

    for col in df_processed.select_dtypes(include=np.number).columns:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed


def main():
    """Orquestra o pipeline completo do projeto."""
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # 1. Carregamento e Preparação dos Dados
    data_dict = load_data()
    df_dev = create_target_variable(data_dict['dev'])
    
    # 2. Combinação e Engenharia de Atributos
    df_dev_full = pd.merge(df_dev, data_dict['cadastral'], on='ID_CLIENTE', how='left')
    df_dev_full = pd.merge(df_dev_full, data_dict['info'], on=ID_COLS, how='left')
    df_dev_featured = feature_engineering(df_dev_full)

    df_teste_full = pd.merge(data_dict['teste'], data_dict['cadastral'], on='ID_CLIENTE', how='left')
    df_teste_full = pd.merge(df_teste_full, data_dict['info'], on=ID_COLS, how='left')
    df_teste_featured = feature_engineering(df_teste_full)
    
    # 3. Pré-processamento e Alinhamento de Colunas
    df_train_processed = preprocess_for_model(df_dev_featured, is_train=True)
    df_test_processed = preprocess_for_model(df_teste_featured, is_train=False)

    train_feature_cols = set(df_train_processed.drop(columns=ID_COLS + [TARGET]).columns)
    test_feature_cols = set(df_test_processed.drop(columns=ID_COLS).columns)
    all_features = sorted(list(train_feature_cols | test_feature_cols))

    X = df_train_processed.reindex(columns=all_features, fill_value=0.0)
    y = df_train_processed[TARGET]
    X_teste = df_test_processed.reindex(columns=all_features, fill_value=0.0)[X.columns]

    # 4. Validação Cruzada com armazenamento de previsões "Out-of-Fold"
    print("\n--- Iniciando Validação Cruzada (5 Folds) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Array para guardar as previsões de validação de cada fold
    oof_preds = np.zeros(len(X))
    auc_scores = []

    # Parâmetros fixos para o modelo
    params = LGBM_FIXED_PARAMS.copy()
    params['learning_rate'] = 0.05

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"--- Fold {fold+1}/5 ---")
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        # Faz a previsão no conjunto de validação do fold atual
        valid_preds = model.predict_proba(X_valid)[:, 1]
        
        # Guarda a previsão no array oof_preds
        oof_preds[val_idx] = valid_preds
        
        auc = roc_auc_score(y_valid, valid_preds)
        auc_scores.append(auc)
        print(f"AUC do Fold {fold+1}: {auc:.4f}")

    # 5. Avaliação Final (usando as previsões out-of-fold)
    print("\n--- Avaliação Final do Modelo (baseada na Validação Cruzada) ---")
    evaluate_model(y, oof_preds, model, X.columns) # Passando y e oof_preds

    # 6. Treino do Modelo Final para Submissão
    final_model = train_final_model(X, y, params)
    test_preds = final_model.predict_proba(X_teste)[:, 1]
    
    # 7. Geração do Arquivo de Submissão
    df_submission = pd.DataFrame({
        'ID_CLIENTE': df_teste_featured['ID_CLIENTE'],
        'SAFRA_REF': df_teste_featured['SAFRA_REF'],
        'PROBABILIDADE_INADIMPLENCIA': test_preds
    })
    df_submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nArquivo de submissão '{SUBMISSION_FILE}' criado com sucesso!")


if __name__ == '__main__':
    main()
