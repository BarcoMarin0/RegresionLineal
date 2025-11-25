# 🤖 Machine Learning Aplicado - Proyecto Final

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)]()

Proyecto integrador de Machine Learning con implementación de modelos supervisados y no supervisados, desplegados en una aplicación web interactiva.

## 📋 Descripción

Este proyecto implementa y compara tres algoritmos de Machine Learning:

- **Regresión Logística** - Predicción de Churn de clientes
- **K-Nearest Neighbors (KNN)** - Clasificación de Churn
- **K-Means Clustering** - Segmentación de clientes

La aplicación permite probar los modelos en tiempo real a través de una interfaz web moderna y responsiva.

## ✨ Características

- 🎯 **3 Modelos de ML Entrenados** con datasets reales
- 📊 **Visualizaciones Interactivas** de métricas y resultados
- 🌐 **Aplicación Web Completa** con FastAPI + React
- 📈 **Comparación de Modelos** con análisis detallado
- 🎨 **UI/UX Moderna** con diseño glassmorphism
- 📱 **Totalmente Responsiva** para todos los dispositivos

## 🛠 Tecnologías

### Backend
- FastAPI 0.110.1
- Scikit-learn 1.7.2
- Pandas 2.3.3
- NumPy 2.3.5
- Matplotlib 3.10.7
- Seaborn 0.13.2

### Frontend
- React 19.0.0
- React Router DOM 7.5.1
- Axios 1.8.4
- Lucide React 0.507.0

## 📦 Instalación

### Prerrequisitos

- Python 3.11+
- Node.js 18+
- Yarn 1.22+

### Backend

```bash
# Navegar al directorio backend
cd backend

# Instalar dependencias
pip install -r requirements.txt

# Generar datasets sintéticos
python3 ../scripts/generate_datasets.py

# Entrenar modelos
python3 ../notebooks/01_regresion_logistica.py
python3 ../notebooks/02_knn.py
python3 ../notebooks/03_kmeans.py

# Iniciar servidor
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend

```bash
# Navegar al directorio frontend
cd frontend

# Instalar dependencias
yarn install

# Iniciar aplicación
yarn start
```

La aplicación estará disponible en `http://localhost:3000`

## 📖 Uso

### Predicción de Churn

1. Acceder a "Regresión Logística" o "K-Nearest Neighbors"
2. Completar el formulario con datos del cliente
3. Click en "Predecir Churn"
4. Ver resultado con probabilidad y nivel de riesgo

### Segmentación de Clientes

1. Acceder a "K-Means Clustering"
2. Ingresar datos financieros del cliente
3. Click en "Identificar Segmento"
4. Ver cluster asignado con descripción del perfil

### Comparación de Modelos

1. Acceder a "Comparación de Modelos"
2. Ver automáticamente:
   - Métricas comparativas
   - Matrices de confusión
   - Conclusiones y recomendaciones

## 📊 Resultados

### Modelos Supervisados

| Modelo                  | Accuracy | Precision | Recall | F1-Score | AUC    |
|------------------------|----------|-----------|--------|----------|--------|
| Regresión Logística    | 67.00%   | 59.77%    | 62.33% | 61.02%   | 0.7451 |
| KNN (K=15)             | 66.08%   | 59.50%    | 56.85% | 58.14%   | 0.7078 |

**Ganador:** Regresión Logística

### Modelo No Supervisado

- **K-Means con K=2**
- **Silhouette Score:** 0.8797
- **Clusters identificados:**
  - Cluster 0: Usuarios de Adelantos en Efectivo (99.7%)
  - Cluster 1: Revolventes de Alto Balance (0.3%)

## 📁 Estructura del Proyecto

```
/app/
├── backend/                    # Backend FastAPI
│   ├── server.py              # Servidor principal
│   ├── routes/
│   │   └── ml_routes.py       # Endpoints ML
│   ├── models/                # Modelos entrenados
│   └── requirements.txt
│
├── frontend/                  # Frontend React
│   ├── src/
│   │   ├── pages/            # Páginas de la app
│   │   └── components/       # Componentes reutilizables
│   └── package.json
│
├── notebooks/                 # Scripts de entrenamiento
│   ├── 01_regresion_logistica.py
│   ├── 02_knn.py
│   └── 03_kmeans.py
│
├── data/                      # Datasets y visualizaciones
│   ├── telco_churn.csv
│   ├── credit_card.csv
│   └── figures/
│
└── DOCUMENTACION_PROYECTO.md  # Documentación completa
```

## 📚 Documentación

Para documentación detallada del proyecto, ver [DOCUMENTACION_PROYECTO.md](DOCUMENTACION_PROYECTO.md)

Incluye:
- Análisis completo de datasets
- Explicación detallada de cada modelo
- Metodología de entrenamiento
- Resultados y conclusiones
- Aplicaciones reales

## 🔗 API Endpoints

### Base URL: `/api/ml`

- `GET /` - Información de la API
- `POST /predict/logistic` - Predicción con Regresión Logística
- `POST /predict/knn` - Predicción con KNN
- `POST /predict/kmeans` - Identificación de cluster
- `GET /metrics` - Métricas de todos los modelos
- `GET /clusters/descriptions` - Descripciones de clusters

Documentación interactiva en: `http://localhost:8001/docs`

## 🎯 Casos de Uso

### Telecomunicaciones
- Identificación de clientes en riesgo de cancelar el servicio
- Estrategias de retención personalizadas
- Optimización de campañas de marketing

### Banca y Finanzas
- Segmentación de clientes para productos específicos
- Gestión de riesgo crediticio
- Programas de fidelización diferenciados
