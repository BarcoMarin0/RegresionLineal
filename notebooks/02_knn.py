#!/usr/bin/env python3
"""
Notebook 2: K-Nearest Neighbors (KNN) para predicción de Churn
Dataset: Telco Customer Churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
import joblib
from pathlib import Path
import json

print("="*60)
print("MODELO 2: K-NEAREST NEIGHBORS (KNN)")
print("Dataset: Telco Customer Churn")
print("="*60)

# ============================================
# 1. CARGA DE DATOS
# ============================================
print("\n1. Cargando datos...")
df = pd.read_csv('/app/data/telco_churn.csv')
print(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# ============================================
# 2. PREPROCESAMIENTO
# ============================================
print("\n2. Preprocesamiento de datos...")

# Eliminar customerID
df_model = df.drop('customerID', axis=1)

# Codificar variable target
df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})

# Identificar columnas categóricas y numéricas
categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('Churn')

# Cargar label encoders del modelo anterior (para consistencia)
label_encoders = joblib.load('/app/backend/models/label_encoders.pkl')

# Codificar variables categóricas
for col in categorical_cols:
    df_model[col] = label_encoders[col].transform(df_model[col])

print("✓ Variables categóricas codificadas")

# Separar features y target
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Split train/test (mismo split que logistic regression)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDatos de entrenamiento: {X_train.shape[0]}")
print(f"Datos de prueba: {X_test.shape[0]}")

# Normalizar datos (importante para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Datos normalizados")

# ============================================
# 3. SELECCIÓN DEL MEJOR K
# ============================================
print("\n3. Buscando el mejor valor de K...")

# Probar diferentes valores de k
k_range = range(3, 21, 2)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    k_scores.append(score)
    print(f"  k={k}: accuracy={score:.4f}")

# Encontrar el mejor k
best_k_idx = np.argmax(k_scores)
best_k = list(k_range)[best_k_idx]
best_score = k_scores[best_k_idx]

print(f"\n✓ Mejor K encontrado: {best_k} (accuracy={best_score:.4f})")

# Graficar K vs Accuracy
fig_dir = Path('/app/data/figures')
fig_dir.mkdir(exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(list(k_range), k_scores, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Mejor K = {best_k}')
plt.xlabel('Número de Vecinos (K)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Selección del Mejor K para KNN', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(fig_dir / 'knn_k_selection.png', dpi=100, bbox_inches='tight')
print("✓ Gráfica de selección de K guardada")
plt.close()

# ============================================
# 4. ENTRENAMIENTO DEL MODELO FINAL
# ============================================
print(f"\n4. Entrenando modelo KNN con K={best_k}...")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

print("✓ Modelo entrenado")

# ============================================
# 5. EVALUACIÓN DEL MODELO
# ============================================
print("\n5. Evaluando modelo...")

# Predicciones
y_pred = knn.predict(X_test_scaled)
y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("MÉTRICAS DE KNN")
print("="*50)
print(f"K:         {best_k}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("="*50)

# Classification Report
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(cm)

# ============================================
# 6. VISUALIZACIONES
# ============================================
print("\n6. Generando visualizaciones...")

# Figura 1: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title(f'Matriz de Confusión - KNN (K={best_k})', fontsize=14, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Predicción', fontsize=12)
plt.xticks([0.5, 1.5], ['No Churn', 'Churn'])
plt.yticks([0.5, 1.5], ['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig(fig_dir / 'knn_confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✓ Matriz de confusión guardada")
plt.close()

# Figura 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#06A77D', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
plt.title(f'Curva ROC - KNN (K={best_k})', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'knn_roc_curve.png', dpi=100, bbox_inches='tight')
print("✓ Curva ROC guardada")
plt.close()

# ============================================
# 7. GUARDAR MODELO Y ARTEFACTOS
# ============================================
print("\n7. Guardando modelo y artefactos...")

model_dir = Path('/app/backend/models')
model_dir.mkdir(exist_ok=True)

# Guardar modelo
joblib.dump(knn, model_dir / 'knn_model.pkl')
print("✓ Modelo guardado")

# Guardar scaler (específico para KNN)
joblib.dump(scaler, model_dir / 'knn_scaler.pkl')
print("✓ Scaler guardado")

# Guardar métricas
metrics = {
    'model': 'K-Nearest Neighbors',
    'k': int(best_k),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc': float(auc),
    'confusion_matrix': cm.tolist(),
    'roc_curve': {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    },
    'k_selection': {
        'k_range': list(k_range),
        'k_scores': k_scores
    }
}

with open(model_dir / 'knn_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✓ Métricas guardadas")

print("\n" + "="*60)
print("✓ ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"\nArchivos generados:")
print(f"  - Modelo: {model_dir / 'knn_model.pkl'}")
print(f"  - Scaler: {model_dir / 'knn_scaler.pkl'}")
print(f"  - Métricas: {model_dir / 'knn_metrics.json'}")
print(f"  - Figuras: {fig_dir}/")
