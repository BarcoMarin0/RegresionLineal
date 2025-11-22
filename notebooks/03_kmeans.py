#!/usr/bin/env python3
"""
Notebook 3: K-Means Clustering para segmentación de clientes
Dataset: Credit Card Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib
from pathlib import Path
import json

print("="*60)
print("MODELO 3: K-MEANS CLUSTERING")
print("Dataset: Credit Card Dataset")
print("="*60)

# ============================================
# 1. CARGA DE DATOS
# ============================================
print("\n1. Cargando datos...")
df = pd.read_csv('/app/data/credit_card.csv')
print(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nPrimeras filas del dataset:")
print(df.head())

print(f"\nInformación del dataset:")
print(df.info())

print(f"\nEstadísticas descriptivas:")
print(df.describe())

# ============================================
# 2. PREPROCESAMIENTO
# ============================================
print("\n2. Preprocesamiento de datos...")

# Eliminar CUST_ID (no es útil para clustering)
df_model = df.drop('CUST_ID', axis=1)

# Verificar valores faltantes
print(f"\nValores faltantes por columna:")
print(df_model.isnull().sum())

# Rellenar valores faltantes con la mediana (si existen)
df_model = df_model.fillna(df_model.median())

print("✓ Valores faltantes manejados")

# Seleccionar features relevantes para clustering
features = [
    'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
    'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
    'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
    'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT'
]

X = df_model[features]

print(f"\nShape de X: {X.shape}")

# Normalizar datos (crucial para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ Datos normalizados")

# ============================================
# 3. SELECCIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS
# ============================================
print("\n3. Determinando el número óptimo de clusters...")

# Método del Codo (Elbow Method)
print("\n  a) Método del Codo...")
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"     k={k}: inertia={kmeans.inertia_:.2f}")

# Método Silhouette
print("\n  b) Método Silhouette...")
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"     k={k}: silhouette score={score:.4f}")

# Encontrar el mejor k basado en silhouette
best_k_idx = np.argmax(silhouette_scores)
best_k = list(k_range)[best_k_idx]
best_silhouette = silhouette_scores[best_k_idx]

print(f"\n✓ Número óptimo de clusters: {best_k} (silhouette={best_silhouette:.4f})")

# ============================================
# 4. VISUALIZACIONES DE SELECCIÓN DE K
# ============================================
print("\n4. Generando visualizaciones de selección de K...")

fig_dir = Path('/app/data/figures')
fig_dir.mkdir(exist_ok=True)

# Figura 1: Método del Codo
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'K seleccionado = {best_k}')
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Inercia (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Método del Codo para K-Means', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(fig_dir / 'kmeans_elbow_method.png', dpi=100, bbox_inches='tight')
print("✓ Método del Codo guardado")
plt.close()

# Figura 2: Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'K seleccionado = {best_k}')
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Análisis Silhouette para K-Means', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(fig_dir / 'kmeans_silhouette_analysis.png', dpi=100, bbox_inches='tight')
print("✓ Análisis Silhouette guardado")
plt.close()

# ============================================
# 5. ENTRENAMIENTO DEL MODELO FINAL
# ============================================
print(f"\n5. Entrenando modelo K-Means con K={best_k}...")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

print("✓ Modelo entrenado")

# Añadir clusters al dataframe
df_model['Cluster'] = cluster_labels

# Contar clientes por cluster
print(f"\nDistribución de clientes por cluster:")
for i in range(best_k):
    count = (cluster_labels == i).sum()
    percentage = count / len(cluster_labels) * 100
    print(f"  Cluster {i}: {count} clientes ({percentage:.1f}%)")

# ============================================
# 6. PERFILAMIENTO DE CLUSTERS
# ============================================
print(f"\n6. Perfilamiento de clusters...")

# Calcular estadísticas por cluster
cluster_profiles = df_model.groupby('Cluster')[features].mean()

print("\nPerfil promedio de cada cluster:")
print(cluster_profiles)

# Crear descripciones de clusters basadas en características
cluster_descriptions = {}

for i in range(best_k):
    profile = cluster_profiles.loc[i]
    
    # Analizar características principales
    balance = profile['BALANCE']
    purchases = profile['PURCHASES']
    cash_advance = profile['CASH_ADVANCE']
    credit_limit = profile['CREDIT_LIMIT']
    purchases_freq = profile['PURCHASES_FREQUENCY']
    
    # Crear descripción
    if purchases > cluster_profiles['PURCHASES'].median() and purchases_freq > 0.5:
        if credit_limit > cluster_profiles['CREDIT_LIMIT'].median():
            desc = "Compradores Frecuentes Premium"
            detail = "Clientes con alto límite de crédito que realizan compras frecuentes y mantienen buenos balances."
        else:
            desc = "Compradores Activos Estándar"
            detail = "Clientes que usan su tarjeta regularmente para compras con límite de crédito moderado."
    elif cash_advance > cluster_profiles['CASH_ADVANCE'].median():
        desc = "Usuarios de Adelantos en Efectivo"
        detail = "Clientes que frecuentemente solicitan adelantos en efectivo, posible riesgo financiero."
    elif balance > cluster_profiles['BALANCE'].median():
        desc = "Revolventes de Alto Balance"
        detail = "Clientes que mantienen balances altos pero compran poco, posiblemente pagando intereses."
    elif purchases < cluster_profiles['PURCHASES'].quantile(0.25):
        desc = "Usuarios Inactivos o Dormidos"
        detail = "Clientes con muy poca actividad en su tarjeta, posibles clientes en riesgo de cancelación."
    else:
        desc = "Usuarios Moderados Equilibrados"
        detail = "Clientes con uso equilibrado de la tarjeta, sin patrones extremos de comportamiento."
    
    cluster_descriptions[i] = {
        'name': desc,
        'description': detail,
        'avg_balance': float(balance),
        'avg_purchases': float(purchases),
        'avg_credit_limit': float(credit_limit),
        'customer_count': int((cluster_labels == i).sum())
    }

print("\n" + "="*70)
print("PERFILES DE CLUSTERS")
print("="*70)
for i, profile in cluster_descriptions.items():
    print(f"\nCluster {i}: {profile['name']}")
    print(f"  Descripción: {profile['description']}")
    print(f"  Balance promedio: ${profile['avg_balance']:.2f}")
    print(f"  Compras promedio: ${profile['avg_purchases']:.2f}")
    print(f"  Límite de crédito promedio: ${profile['avg_credit_limit']:.2f}")
    print(f"  Número de clientes: {profile['customer_count']}")

# ============================================
# 7. VISUALIZACIONES DE CLUSTERS
# ============================================
print("\n7. Generando visualizaciones de clusters...")

# Figura 1: Distribución de clusters
plt.figure(figsize=(10, 6))
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Número de Clientes', fontsize=12)
plt.title('Distribución de Clientes por Cluster', fontsize=14, fontweight='bold')
for i, v in enumerate(cluster_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'kmeans_cluster_distribution.png', dpi=100, bbox_inches='tight')
print("✓ Distribución de clusters guardada")
plt.close()

# Figura 2: Heatmap de perfiles de clusters
plt.figure(figsize=(14, 8))
cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
sns.heatmap(cluster_profiles_norm.T, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Valor Normalizado'})
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Heatmap de Perfiles de Clusters (Valores Normalizados)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'kmeans_cluster_heatmap.png', dpi=100, bbox_inches='tight')
print("✓ Heatmap de clusters guardado")
plt.close()

# Figura 3: Scatter plot 2D (usando PCA para reducción de dimensionalidad)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='X', s=300, c='red', edgecolors='black', linewidths=2, label='Centroides')
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title('Visualización de Clusters (PCA 2D)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'kmeans_cluster_visualization.png', dpi=100, bbox_inches='tight')
print("✓ Visualización de clusters guardada")
plt.close()

# ============================================
# 8. GUARDAR MODELO Y ARTEFACTOS
# ============================================
print("\n8. Guardando modelo y artefactos...")

model_dir = Path('/app/backend/models')
model_dir.mkdir(exist_ok=True)

# Guardar modelo
joblib.dump(kmeans, model_dir / 'kmeans_model.pkl')
print("✓ Modelo guardado")

# Guardar scaler
joblib.dump(scaler, model_dir / 'kmeans_scaler.pkl')
print("✓ Scaler guardado")

# Guardar nombres de features
joblib.dump(features, model_dir / 'kmeans_feature_names.pkl')
print("✓ Nombres de features guardados")

# Guardar descripciones de clusters
with open(model_dir / 'cluster_descriptions.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_descriptions, f, indent=2, ensure_ascii=False)
print("✓ Descripciones de clusters guardadas")

# Guardar métricas
metrics = {
    'model': 'K-Means',
    'n_clusters': int(best_k),
    'silhouette_score': float(best_silhouette),
    'inertia': float(kmeans.inertia_),
    'cluster_sizes': {int(i): int((cluster_labels == i).sum()) for i in range(best_k)},
    'elbow_analysis': {
        'k_range': list(k_range),
        'inertias': inertias
    },
    'silhouette_analysis': {
        'k_range': list(k_range),
        'scores': silhouette_scores
    }
}

with open(model_dir / 'kmeans_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✓ Métricas guardadas")

print("\n" + "="*60)
print("✓ ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"\nArchivos generados:")
print(f"  - Modelo: {model_dir / 'kmeans_model.pkl'}")
print(f"  - Scaler: {model_dir / 'kmeans_scaler.pkl'}")
print(f"  - Feature Names: {model_dir / 'kmeans_feature_names.pkl'}")
print(f"  - Cluster Descriptions: {model_dir / 'cluster_descriptions.json'}")
print(f"  - Métricas: {model_dir / 'kmeans_metrics.json'}")
print(f"  - Figuras: {fig_dir}/")

print("\n" + "="*60)
print("APLICACIONES REALES DEL CLUSTERING")
print("="*60)
print("""
1. Segmentación de Marketing: 
   - Diseñar campañas personalizadas para cada segmento de clientes
   - Ofrecer productos y servicios específicos según el perfil

2. Gestión de Riesgo:
   - Identificar clientes con alto riesgo (usuarios de adelantos frecuentes)
   - Ajustar límites de crédito según comportamiento

3. Retención de Clientes:
   - Detectar clientes inactivos y crear estrategias de reactivación
   - Identificar clientes valiosos para programas de fidelización

4. Optimización de Productos:
   - Desarrollar productos específicos para cada segmento
   - Ajustar tasas de interés y beneficios según perfil

5. Predicción de Comportamiento:
   - Anticipar necesidades financieras de cada segmento
   - Predecir qué clientes podrían necesitar productos adicionales
""")
