import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

LABELS = ['A', 'D', 'H']

def evaluar_modelo(y_true, y_pred_class, y_pred_proba, nombre_modelo, verbose=True):
    acc = accuracy_score(y_true, y_pred_class)
    f1_macro = f1_score(y_true, y_pred_class, labels=LABELS, average='macro')
    f1_por_clase = f1_score(y_true, y_pred_class, labels=LABELS, average=None)
    ll = log_loss(y_true, y_pred_proba, labels=LABELS)
    cm = confusion_matrix(y_true, y_pred_class, labels=LABELS)
    
    if verbose:
        print(f"=== {nombre_modelo} ===")
        print(f"  Accuracy     : {acc:.4f}")
        print(f"  F1-macro     : {f1_macro:.4f}")
        print(f"  F1 por clase : A={f1_por_clase[0]:.4f}  D={f1_por_clase[1]:.4f}  H={f1_por_clase[2]:.4f}")
        print(f"  Log-loss     : {ll:.4f}")
        print(f"\n  Matriz de confusión (filas=verdadero, columnas=predicho, orden A/D/H):")
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
        print(cm_df.to_string())
        print()
    
    return {
        'modelo': nombre_modelo,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_A': f1_por_clase[0],
        'f1_D': f1_por_clase[1],
        'f1_H': f1_por_clase[2],
        'log_loss': ll,
        'confusion_matrix': cm.tolist(),
    }
    
    
def graficar_matriz_confusion(cm, labels, title, ax=None):
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f"Matriz de confusión · {title}")
    ax.set_ylabel('Etiqueta verdadera')
    ax.set_xlabel('Etiqueta predicha')
    return ax
    
def comparar_modelos(resultados, ordenar_por='log_loss', ascendente=True):
    df = pd.DataFrame(resultados)
    columnas = ['modelo', 'accuracy', 'f1_macro', 'f1_A', 'f1_D', 'f1_H', 'log_loss']
    df = df[columnas].sort_values(ordenar_por, ascending=ascendente).reset_index(drop=True)
    return df