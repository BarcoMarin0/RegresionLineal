#!/usr/bin/env python3
"""
Notebook 1: Regresión Logística para predicción de Churn
Dataset: Telco Customer Churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
import joblib
from pathlib import Path
import json

print("="*60)
print("MODELO 1: REGRESIÓN LOGÍSTICA")
print("Dataset: Telco Customer Churn")
print("="*60)

# ============================================
# 1. CARGA DE DATOS
# ============================================
print("\n1. Cargando datos...")
df = pd.read_csv('/app/data/telco_churn.csv')
print(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nPrimeras filas del dataset:")
print(df.head())

print(f"\nInformación del dataset:")
print(df.info())

print(f"\nDistribución de Churn:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))

# ============================================
# 2. PREPROCESAMIENTO
# ============================================
print("\n2. Preprocesamiento de datos...")

# Eliminar customerID (no es útil para predicción)
df_model = df.drop('customerID', axis=1)

# Codificar variable target
df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})

# Identificar columnas categóricas y numéricas
categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('Churn')  # Excluir target

print(f"Columnas categóricas: {len(categorical_cols)}")
print(f"Columnas numéricas: {len(numeric_cols)}")

# Codificar variables categóricas usando Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

print("✓ Variables categóricas codificadas")

# Separar features y target
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

print(f"\nShape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDatos de entrenamiento: {X_train.shape[0]}")
print(f"Datos de prueba: {X_test.shape[0]}")

# Normalizar datos numéricos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Datos normalizados")

# ============================================
# 3. ENTRENAMIENTO DEL MODELO
# ============================================
print("\n3. Entrenando modelo de Regresión Logística...")

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

print("✓ Modelo entrenado")

# ============================================
# 4. EVALUACIÓN DEL MODELO
# ============================================
print("\n4. Evaluando modelo...")

# Predicciones
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("MÉTRICAS DE REGRESIÓN LOGÍSTICA")
print("="*50)
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
# 5. VISUALIZACIONES
# ============================================
print("\n5. Generando visualizaciones...")

# Crear directorio para gráficas
fig_dir = Path('/app/data/figures')
fig_dir.mkdir(exist_ok=True)

# Figura 1: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - Regresión Logística', fontsize=14, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Predicción', fontsize=12)
plt.xticks([0.5, 1.5], ['No Churn', 'Churn'])
plt.yticks([0.5, 1.5], ['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig(fig_dir / 'logistic_confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✓ Matriz de confusión guardada")
plt.close()

# Figura 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2E86AB', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
plt.title('Curva ROC - Regresión Logística', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'logistic_roc_curve.png', dpi=100, bbox_inches='tight')
print("✓ Curva ROC guardada")
plt.close()

# Figura 3: Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(log_reg.coef_[0])
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
plt.title('Top 15 Features Más Importantes - Regresión Logística', fontsize=14, fontweight='bold')
plt.xlabel('Importancia (|Coeficiente|)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(fig_dir / 'logistic_feature_importance.png', dpi=100, bbox_inches='tight')
print("✓ Importancia de features guardada")
plt.close()

# ============================================
# 6. GUARDAR MODELO Y ARTEFACTOS
# ============================================
print("\n6. Guardando modelo y artefactos...")

model_dir = Path('/app/backend/models')
model_dir.mkdir(exist_ok=True)

# Guardar modelo
joblib.dump(log_reg, model_dir / 'logistic_regression_model.pkl')
print("✓ Modelo guardado")

# Guardar scaler
joblib.dump(scaler, model_dir / 'logistic_scaler.pkl')
print("✓ Scaler guardado")

# Guardar label encoders
joblib.dump(label_encoders, model_dir / 'label_encoders.pkl')
print("✓ Label encoders guardados")

# Guardar nombres de features
joblib.dump(X.columns.tolist(), model_dir / 'feature_names.pkl')
print("✓ Nombres de features guardados")

# Guardar métricas
metrics = {
    'model': 'Logistic Regression',
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc': float(auc),
    'confusion_matrix': cm.tolist(),
    'roc_curve': {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
}

with open(model_dir / 'logistic_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✓ Métricas guardadas")

print("\n" + "="*60)
print("✓ ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"\nArchivos generados:")
print(f"  - Modelo: {model_dir / 'logistic_regression_model.pkl'}")
print(f"  - Scaler: {model_dir / 'logistic_scaler.pkl'}")
print(f"  - Label Encoders: {model_dir / 'label_encoders.pkl'}")
print(f"  - Feature Names: {model_dir / 'feature_names.pkl'}")
print(f"  - Métricas: {model_dir / 'logistic_metrics.json'}")
print(f"  - Figuras: {fig_dir}/")
