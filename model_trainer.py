import lightgbm as lgb
#import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from config import LGBM_FIXED_PARAMS, OPTUNA_TRIALS


def train_model_on_fold(X_train, y_train, X_valid, y_valid, params):
    """Treina e avalia um modelo em um único fold da validação cruzada."""
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(50, verbose=False)])
    
    valid_preds = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, valid_preds), model

def train_final_model(X, y, params):
    """Treina o modelo final com todos os dados."""
    print("\n--- Treinando Modelo Final com Todos os Dados ---")
    final_params = params.copy()
    final_params['n_estimators'] = 3000 
    
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X, y)
    return model
