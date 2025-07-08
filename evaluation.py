import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
from config import OUTPUT_PATH

def evaluate_model(y_true, y_pred_proba, model, features):
    """Calcula métricas, gera e salva gráficos de avaliação."""
    print("\n--- Avaliação do Modelo ---")
    
    # 1. AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"AUC na validação: {auc:.4f}")

    # 2. Relatório de Classificação (limiar de 0.5)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    print("\nRelatório de Classificação (limiar=0.5):")
    print(classification_report(y_true, y_pred_binary))

    # 3. Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Em Dia', 'Inadimplente'], yticklabels=['Em Dia', 'Inadimplente'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.savefig(f"{OUTPUT_PATH}confusion_matrix.png")
    plt.close()
    print("Matriz de confusão salva em 'output/'.")

    # 4. Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Curva Precision-Recall')
    plt.xlabel('Recall (Sensibilidade)')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_PATH}precision_recall_curve.png")
    plt.close()
    print("Curva Precision-Recall salva em 'output/'.")
    
    # 5. Importância das Features
    plt.figure(figsize=(10, 12))
    sns.barplot(x=model.feature_importances_, y=features, orient='h', order=features[model.feature_importances_.argsort()[::-1]][:30])
    plt.title("Importância das Features (Top 30)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}feature_importance.png")
    plt.close()
    print("Gráfico de importância das features salvo em 'output/'.")
