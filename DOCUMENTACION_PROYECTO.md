## Examen Final - Proyecto Integrador
---

## 🗂 Estructura del Proyecto

```
/app/
├── backend/                    # Backend FastAPI
│   ├── server.py              # Servidor principal
│   ├── routes/
│   │   └── ml_routes.py       # Endpoints de ML
│   ├── models/                # Modelos entrenados (.pkl)
│   │   ├── logistic_regression_model.pkl
│   │   ├── logistic_scaler.pkl
│   │   ├── knn_model.pkl
│   │   ├── knn_scaler.pkl
│   │   ├── kmeans_model.pkl
│   │   ├── kmeans_scaler.pkl
│   │   ├── label_encoders.pkl
│   │   ├── feature_names.pkl
│   │   ├── kmeans_feature_names.pkl
│   │   ├── cluster_descriptions.json
│   │   ├── logistic_metrics.json
│   │   ├── knn_metrics.json
│   │   └── kmeans_metrics.json
│   └── requirements.txt
│
├── frontend/                  # Frontend React
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── pages/
│   │   │   ├── Home.js
│   │   │   ├── LogisticRegression.js
│   │   │   ├── KNN.js
│   │   │   ├── KMeans.js
│   │   │   └── Comparacion.js
│   │   └── components/
│   │       ├── Navigation.js
│   │       └── ResultCard.js
│   └── package.json
│
├── notebooks/                 # Scripts de entrenamiento
│   ├── 01_regresion_logistica.py
│   ├── 02_knn.py
│   └── 03_kmeans.py
│
├── data/                      # Datos y visualizaciones
│   ├── telco_churn.csv
│   ├── credit_card.csv
│   └── figures/               # Gráficas generadas
│       ├── logistic_confusion_matrix.png
│       ├── logistic_roc_curve.png
│       ├── logistic_feature_importance.png
│       ├── knn_confusion_matrix.png
│       ├── knn_roc_curve.png
│       ├── knn_k_selection.png
│       ├── kmeans_elbow_method.png
│       ├── kmeans_silhouette_analysis.png
│       ├── kmeans_cluster_distribution.png
│       ├── kmeans_cluster_heatmap.png
│       └── kmeans_cluster_visualization.png
│
└── scripts/
    └── generate_datasets.py   # Generación de datasets sintéticos
```

---

## 📊 Datasets Utilizados

### 1. Telco Customer Churn (Modelos Supervisados)

**Descripción:** Dataset de clientes de telecomunicaciones para predicción de Churn.

**Target:** Churn (Yes/No)

**Características:**
- **Registros:** 7,043 clientes
- **Distribución de Churn:**
  - No Churn: 58.6% (4,125 clientes)
  - Churn: 41.4% (2,918 clientes)

**Variables:**
- **Demográficas:** gender, SeniorCitizen, Partner, Dependents
- **Servicios:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Contractuales:** tenure, Contract, PaperlessBilling, PaymentMethod
- **Financieras:** MonthlyCharges, TotalCharges

### 2. Credit Card Dataset (Modelo No Supervisado)

**Descripción:** Dataset de comportamiento de tarjetas de crédito para segmentación de clientes.

**Características:**
- **Registros:** 8,950 clientes
- **Balance Promedio:** $5,591.50
- **Compras Promedio:** $3,212.19

**Variables:**
- BALANCE, PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES
- CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS
- PURCHASES_FREQUENCY, ONEOFF_PURCHASES_FREQUENCY
- PURCHASES_INSTALLMENTS_FREQUENCY, CASH_ADVANCE_FREQUENCY
- CASH_ADVANCE_TRX, PURCHASES_TRX, PRC_FULL_PAYMENT

---

## 🤖 Modelos Implementados

### 1. Regresión Logística

**Propósito:** Predicción de probabilidad de Churn de clientes.

**Preprocesamiento:**
- Label Encoding para variables categóricas
- StandardScaler para normalización de features
- Train-Test Split: 80/20 con estratificación

**Hiperparámetros:**
- `random_state=42`
- `max_iter=1000`

**Resultados:**
```
Accuracy:   67.00%
Precision:  59.77%
Recall:     62.33%
F1-Score:   61.02%
AUC:        0.7451
```

**Matriz de Confusión:**
```
                 Predicción
              No Churn  Churn
Real No Churn    580      245
     Churn       220      364
```

**Interpretación:**
- El modelo identifica correctamente el 70% de los clientes que no harán churn
- Captura el 62% de los clientes que realmente harán churn
- La curva ROC muestra un desempeño significativamente mejor que un clasificador aleatorio (AUC = 0.7451)

### 2. K-Nearest Neighbors (KNN)

**Propósito:** Clasificación de Churn basada en similitud con vecinos cercanos.

**Preprocesamiento:**
- Mismo encoding que Regresión Logística
- StandardScaler (crucial para KNN por sensibilidad a escalas)

**Selección de K:**
- Rango evaluado: K = 3 a 19 (valores impares)
- **Mejor K encontrado: 15**
- Método: maximización de accuracy en set de validación

**Resultados:**
```
K:          15
Accuracy:   66.08%
Precision:  59.50%
Recall:     56.85%
F1-Score:   58.14%
AUC:        0.7078
```

**Matriz de Confusión:**
```
                 Predicción
              No Churn  Churn
Real No Churn    599      226
     Churn       252      332
```

**Interpretación:**
- Similar accuracy a Regresión Logística pero con menor recall
- Mejor en predecir "No Churn" (72.6% de precisión)
- El valor K=15 proporciona un buen balance entre sesgo y varianza

### 3. K-Means Clustering

**Propósito:** Segmentación de clientes de tarjetas de crédito en grupos homogéneos.

**Preprocesamiento:**
- Manejo de valores faltantes con mediana
- StandardScaler para todas las features numéricas
- 15 features utilizadas para clustering

**Selección del Número Óptimo de Clusters:**

**Método del Codo (Elbow Method):**
- Evaluó K de 2 a 10
- Inercia disminuye de 115,067 (K=2) a 58,673 (K=10)
- Codo suave, no muy pronunciado

**Método Silhouette:**
- **K=2: Score = 0.8797** ← Óptimo seleccionado
- K=3: Score = 0.2037
- K=4: Score = 0.2086

**Número Óptimo: K=2**

**Resultados:**
```
Número de Clusters: 2
Silhouette Score:   0.8797
Inercia:            115,067.53
```

**Distribución de Clusters:**
- **Cluster 0:** 8,925 clientes (99.7%)
- **Cluster 1:** 25 clientes (0.3%)

**Perfilamiento de Clusters:**

**Cluster 0: Usuarios de Adelantos en Efectivo**
- Balance Promedio: $5,332.32
- Compras Promedio: $2,550.35
- Límite de Crédito Promedio: $14,685.68
- Clientes: 8,925
- **Descripción:** Clientes que frecuentemente solicitan adelantos en efectivo, posible riesgo financiero.

**Cluster 1: Revolventes de Alto Balance**
- Balance Promedio: $98,119.44
- Compras Promedio: $239,489.41
- Límite de Crédito Promedio: $259,918.95
- Clientes: 25
- **Descripción:** Clientes que mantienen balances altos pero compran poco, posiblemente pagando intereses. Clientes VIP con alto poder adquisitivo.

**Aplicaciones Reales:**

1. **Segmentación de Marketing:**
   - Diseñar campañas personalizadas para cada segmento
   - Ofrecer productos específicos según perfil

2. **Gestión de Riesgo:**
   - Identificar clientes con alto riesgo (Cluster 0)
   - Ajustar límites de crédito según comportamiento

3. **Retención de Clientes:**
   - Detectar clientes valiosos (Cluster 1)
   - Programas de fidelización diferenciados

4. **Optimización de Productos:**
   - Desarrollar productos específicos por segmento
   - Ajustar tasas de interés según perfil

5. **Predicción de Comportamiento:**
   - Anticipar necesidades financieras
   - Predecir necesidad de productos adicionales

---

## 📈 Análisis y Resultados

### Comparación de Modelos Supervisados

| Métrica    | Regresión Logística | KNN (K=15) | Ganador              |
|------------|---------------------|------------|----------------------|
| Accuracy   | **67.00%**          | 66.08%     | Regresión Logística  |
| Precision  | **59.77%**          | 59.50%     | Regresión Logística  |
| Recall     | **62.33%**          | 56.85%     | Regresión Logística  |
| F1-Score   | **61.02%**          | 58.14%     | Regresión Logística  |
| AUC        | **0.7451**          | 0.7078     | Regresión Logística  |

### Conclusiones del Análisis Comparativo

**Ganador General: Regresión Logística**

**Razones:**
1. **Mayor Accuracy:** 67.00% vs 66.08% del KNN
2. **Mejor AUC:** 0.7451 indica mejor capacidad de discriminación
3. **Recall Superior:** Captura más casos positivos de Churn (62.33% vs 56.85%)
4. **Interpretabilidad:** Proporciona coeficientes interpretables y probabilidades calibradas

**Cuándo usar cada modelo:**

**Regresión Logística:**
- ✅ Cuando se requiere interpretabilidad de resultados
- ✅ Necesidad de probabilidades calibradas
- ✅ Relaciones lineales entre features y target
- ✅ Recomendado para este caso de Churn

**KNN:**
- ✅ Relaciones no lineales complejas
- ✅ Modelo flexible sin suposiciones de distribución
- ✅ Datos con fronteras de decisión irregulares
- ✅ Útil como modelo de comparación

---

## 🌐 Aplicación Web

### Tecnologías Frontend

- **Framework:** React 19.0.0
- **Routing:** React Router DOM 7.5.1
- **Estilos:** CSS3 con diseño moderno y responsivo
- **Iconos:** Lucide React
- **HTTP Client:** Axios 1.8.4

### Tecnologías Backend

- **Framework:** FastAPI 0.110.1
- **Machine Learning:** Scikit-learn 1.7.2
- **Serialización de Modelos:** Joblib 1.5.2
- **Validación de Datos:** Pydantic 2.6.4
- **Servidor:** Uvicorn 0.25.0

### Arquitectura de la Aplicación

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   React     │  HTTP   │   FastAPI   │  Load   │   Modelos   │
│  Frontend   │ ─────>  │   Backend   │ ─────>  │  ML (.pkl)  │
│             │ <─────  │             │ <─────  │             │
└─────────────┘  JSON   └─────────────┘ Predict └─────────────┘
```

### Endpoints de la API

**Base URL:** `/api/ml`

#### 1. GET `/api/ml/`
- **Descripción:** Información general de la API
- **Respuesta:**
```json
{
  "message": "API de Machine Learning",
  "models": [
    "Regresión Logística (Churn)",
    "K-Nearest Neighbors (Churn)",
    "K-Means Clustering (Segmentación)"
  ]
}
```

#### 2. POST `/api/ml/predict/logistic`
- **Descripción:** Predicción de Churn con Regresión Logística
- **Request Body:** Datos del cliente (TelcoCustomerInput)
- **Respuesta:**
```json
{
  "prediction": "Yes",
  "probability": 0.7224,
  "confidence": "Alto riesgo de Churn"
}
```

#### 3. POST `/api/ml/predict/knn`
- **Descripción:** Predicción de Churn con KNN
- **Request Body:** Datos del cliente (TelcoCustomerInput)
- **Respuesta:** Mismo formato que Regresión Logística

#### 4. POST `/api/ml/predict/kmeans`
- **Descripción:** Identificación de cluster
- **Request Body:** Datos financieros (CreditCardInput)
- **Respuesta:**
```json
{
  "cluster": 0,
  "cluster_name": "Usuarios de Adelantos en Efectivo",
  "cluster_description": "Clientes que frecuentemente solicitan adelantos...",
  "avg_balance": 5332.32,
  "avg_purchases": 2550.35,
  "avg_credit_limit": 14685.68,
  "customer_count": 8925
}
```

#### 5. GET `/api/ml/metrics`
- **Descripción:** Obtiene métricas de todos los modelos
- **Respuesta:** JSON con métricas completas de cada modelo

### Páginas de la Aplicación

#### 1. Home (`/`)
- **Función:** Página principal con navegación
- **Características:**
  - 4 tarjetas interactivas para cada modelo/sección
  - Diseño moderno con gradientes y animaciones
  - Iconos representativos de cada modelo

#### 2. Regresión Logística (`/logistic`)
- **Función:** Formulario de predicción con Regresión Logística
- **Secciones:**
  - Información Personal
  - Servicios
  - Facturación y Pagos
- **Resultado:**
  - Predicción (Yes/No)
  - Probabilidad de Churn
  - Barra de progreso visual
  - Nivel de confianza

#### 3. K-Nearest Neighbors (`/knn`)
- **Función:** Formulario de predicción con KNN
- **Características:** Mismo formulario que Regresión Logística
- **Diferencia:** Usa el modelo KNN con K=15 vecinos

#### 4. K-Means Clustering (`/kmeans`)
- **Función:** Identificación de segmento de cliente
- **Secciones:**
  - Datos Financieros
  - Frecuencias de Uso
  - Transacciones
- **Resultado:**
  - Número de cluster asignado
  - Nombre del segmento
  - Descripción detallada
  - Estadísticas del cluster

#### 5. Comparación de Modelos (`/comparacion`)
- **Función:** Análisis comparativo de modelos supervisados
- **Contenido:**
  - Tabla comparativa de métricas
  - Matrices de confusión lado a lado
  - Conclusiones automáticas
  - Recomendaciones de uso

### Características de Diseño

**Principios de Diseño Aplicados:**
- ✨ Diseño moderno con glassmorphism
- 🎨 Paleta de colores distintiva por modelo
- 📱 Totalmente responsivo
- ⚡ Animaciones fluidas y transiciones
- 🎯 Navegación intuitiva
- 📊 Visualización clara de resultados

**Tipografía:**
- Encabezados: Space Grotesk (distintivo y moderno)
- Texto: Inter (legible y profesional)

**Colores por Modelo:**
- Regresión Logística: Azul (#4f9cf9)
- KNN: Verde (#06a77d)
- K-Means: Naranja (#f59e0b)
- Comparación: Púrpura (#8b5cf6)

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.11+
- Node.js 18+
- Yarn 1.22+

### Instalación del Backend

```bash
cd /app/backend

# Instalar dependencias
pip install -r requirements.txt

# Generar datasets sintéticos
python3 /app/scripts/generate_datasets.py

# Entrenar modelos
python3 /app/notebooks/01_regresion_logistica.py
python3 /app/notebooks/02_knn.py
python3 /app/notebooks/03_kmeans.py

# Iniciar servidor
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Instalación del Frontend

```bash
cd /app/frontend

# Instalar dependencias
yarn install

# Configurar variables de entorno
# Editar .env con REACT_APP_BACKEND_URL

# Iniciar aplicación
yarn start
```

### Variables de Entorno

**Backend (`.env`):**
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=test_database
CORS_ORIGINS=*
```

**Frontend (`.env`):**
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

---

## 💻 Uso de la Aplicación

### Predicción de Churn (Regresión Logística / KNN)

1. Acceder a `/logistic` o `/knn`
2. Completar el formulario con datos del cliente:
   - Información personal (género, edad, dependientes)
   - Servicios contratados (internet, TV, soporte)
   - Información contractual (tipo de contrato, método de pago)
   - Datos financieros (cargos mensuales y totales)
3. Click en "Predecir Churn" o "Predecir Churn con KNN"
4. Ver resultado:
   - Predicción (Cliente en Riesgo / Cliente Estable)
   - Probabilidad de Churn (0-100%)
   - Nivel de confianza (Bajo/Moderado/Alto riesgo)

### Segmentación de Clientes (K-Means)

1. Acceder a `/kmeans`
2. Ingresar datos financieros:
   - Balances y compras
   - Adelantos en efectivo
   - Límite de crédito
   - Frecuencias de uso
   - Número de transacciones
3. Click en "Identificar Segmento"
4. Ver resultado:
   - Cluster asignado (0 o 1)
   - Nombre del segmento
   - Descripción del perfil
   - Estadísticas comparativas

### Comparación de Modelos

1. Acceder a `/comparacion`
2. Ver automáticamente:
   - Tabla comparativa de métricas
   - Matrices de confusión
   - Identificación del mejor modelo
   - Conclusiones y recomendaciones

---

## 🛠 Tecnologías Utilizadas

### Backend
- **FastAPI:** Framework web moderno y rápido
- **Scikit-learn:** Biblioteca de Machine Learning
- **Pandas:** Manipulación de datos
- **NumPy:** Computación numérica
- **Matplotlib:** Visualización de datos
- **Seaborn:** Visualización estadística
- **Joblib:** Serialización de modelos
- **Pydantic:** Validación de datos

### Frontend
- **React:** Biblioteca de UI
- **React Router:** Navegación
- **Axios:** Cliente HTTP
- **Lucide React:** Iconos modernos
- **CSS3:** Estilos y animaciones

### Herramientas de Desarrollo
- **Python 3.11:** Lenguaje del backend
- **JavaScript ES6+:** Lenguaje del frontend
- **Git:** Control de versiones

---

### Bibliotecas y Frameworks
- Scikit-learn Documentation: https://scikit-learn.org/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- React Documentation: https://react.dev/

### Papers y Artículos
- Logistic Regression for Machine Learning
- K-Nearest Neighbors Algorithm
- K-Means Clustering Analysis

---
