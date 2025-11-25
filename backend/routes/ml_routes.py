""" 
Rutas para los modelos de Machine Learning
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import numpy as np
import json
from pathlib import Path

# Router para ML
ml_router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Directorio de modelos
import os
MODEL_DIR = Path(os.getcwd()) / "models"


# ============================================
# MODELOS Y ARTEFACTOS
# ============================================

# Cargar modelos
logistic_model = joblib.load(MODEL_DIR / 'logistic_regression_model.pkl')
logistic_scaler = joblib.load(MODEL_DIR / 'logistic_scaler.pkl')

knn_model = joblib.load(MODEL_DIR / 'knn_model.pkl')
knn_scaler = joblib.load(MODEL_DIR / 'knn_scaler.pkl')

kmeans_model = joblib.load(MODEL_DIR / 'kmeans_model.pkl')
kmeans_scaler = joblib.load(MODEL_DIR / 'kmeans_scaler.pkl')

# Cargar artefactos adicionales
label_encoders = joblib.load(MODEL_DIR / 'label_encoders.pkl')
feature_names = joblib.load(MODEL_DIR / 'feature_names.pkl')
kmeans_feature_names = joblib.load(MODEL_DIR / 'kmeans_feature_names.pkl')

# Cargar métricas
with open(MODEL_DIR / 'logistic_metrics.json', 'r') as f:
    logistic_metrics = json.load(f)

with open(MODEL_DIR / 'knn_metrics.json', 'r') as f:
    knn_metrics = json.load(f)

with open(MODEL_DIR / 'kmeans_metrics.json', 'r') as f:
    kmeans_metrics = json.load(f)

with open(MODEL_DIR / 'cluster_descriptions.json', 'r', encoding='utf-8') as f:
    cluster_descriptions = json.load(f)

# ============================================
# SCHEMAS PYDANTIC
# ============================================

class TelcoCustomerInput(BaseModel):
    """Datos de entrada para predicción de Churn"""
    gender: str = Field(..., description="Male o Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 o 1")
    Partner: str = Field(..., description="Yes o No")
    Dependents: str = Field(..., description="Yes o No")
    tenure: int = Field(..., ge=0, le=72, description="Meses de antigüedad (0-72)")
    PhoneService: str = Field(..., description="Yes o No")
    MultipleLines: str = Field(..., description="Yes, No, o No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, o No")
    OnlineSecurity: str = Field(..., description="Yes, No, o No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, o No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, o No internet service")
    TechSupport: str = Field(..., description="Yes, No, o No internet service")
    StreamingTV: str = Field(..., description="Yes, No, o No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, o No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, o Two year")
    PaperlessBilling: str = Field(..., description="Yes o No")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), o Credit card (automatic)")
    MonthlyCharges: float = Field(..., ge=0, description="Cargos mensuales")
    TotalCharges: float = Field(..., ge=0, description="Cargos totales")

class ChurnPredictionResponse(BaseModel):
    """Respuesta de predicción de Churn"""
    prediction: str = Field(..., description="Yes o No")
    probability: float = Field(..., description="Probabilidad de Churn")
    confidence: str = Field(..., description="Nivel de confianza")

class CreditCardInput(BaseModel):
    """Datos de entrada para clustering de tarjetas de crédito"""
    BALANCE: float = Field(..., description="Balance de la cuenta")
    PURCHASES: float = Field(..., description="Monto de compras")
    ONEOFF_PURCHASES: float = Field(..., description="Compras únicas")
    INSTALLMENTS_PURCHASES: float = Field(..., description="Compras a plazos")
    CASH_ADVANCE: float = Field(..., description="Adelantos en efectivo")
    PURCHASES_FREQUENCY: float = Field(..., ge=0, le=1, description="Frecuencia de compras (0-1)")
    ONEOFF_PURCHASES_FREQUENCY: float = Field(..., ge=0, le=1, description="Frecuencia de compras únicas (0-1)")
    PURCHASES_INSTALLMENTS_FREQUENCY: float = Field(..., ge=0, le=1, description="Frecuencia de compras a plazos (0-1)")
    CASH_ADVANCE_FREQUENCY: float = Field(..., ge=0, le=1, description="Frecuencia de adelantos (0-1)")
    CASH_ADVANCE_TRX: int = Field(..., ge=0, description="Número de transacciones de adelantos")
    PURCHASES_TRX: int = Field(..., ge=0, description="Número de transacciones de compras")
    CREDIT_LIMIT: float = Field(..., description="Límite de crédito")
    PAYMENTS: float = Field(..., description="Monto de pagos")
    MINIMUM_PAYMENTS: float = Field(..., description="Pagos mínimos")
    PRC_FULL_PAYMENT: float = Field(..., ge=0, le=1, description="Porcentaje de pago completo (0-1)")

class ClusterPredictionResponse(BaseModel):
    """Respuesta de predicción de cluster"""
    cluster: int = Field(..., description="Número del cluster")
    cluster_name: str = Field(..., description="Nombre del cluster")
    cluster_description: str = Field(..., description="Descripción del cluster")
    avg_balance: float = Field(..., description="Balance promedio del cluster")
    avg_purchases: float = Field(..., description="Compras promedio del cluster")
    avg_credit_limit: float = Field(..., description="Límite de crédito promedio del cluster")
    customer_count: int = Field(..., description="Número de clientes en el cluster")

class MetricsResponse(BaseModel):
    """Métricas de los modelos"""
    logistic_regression: Dict
    knn: Dict
    kmeans: Dict

# ============================================
# ENDPOINTS
# ============================================

@ml_router.get("/")
async def ml_root():
    """Endpoint raíz de ML"""
    return {
        "message": "API de Machine Learning",
        "models": [
            "Regresión Logística (Churn)",
            "K-Nearest Neighbors (Churn)",
            "K-Means Clustering (Segmentación)"
        ]
    }

@ml_router.post("/predict/logistic", response_model=ChurnPredictionResponse)
async def predict_churn_logistic(customer: TelcoCustomerInput):
    """
    Predice Churn usando Regresión Logística
    """
    try:
        # Convertir input a diccionario
        customer_dict = customer.model_dump()
        
        # Codificar variables categóricas
        for col, encoder in label_encoders.items():
            if col in customer_dict:
                try:
                    customer_dict[col] = encoder.transform([customer_dict[col]])[0]
                except ValueError:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Valor inválido para {col}: {customer_dict[col]}"
                    )
        
        # Crear array con el orden correcto de features
        features = np.array([customer_dict[col] for col in feature_names]).reshape(1, -1)
        
        # Normalizar
        features_scaled = logistic_scaler.transform(features)
        
        # Predecir
        prediction = logistic_model.predict(features_scaled)[0]
        probability = logistic_model.predict_proba(features_scaled)[0][1]
        
        # Determinar nivel de confianza
        if probability < 0.3:
            confidence = "Bajo riesgo de Churn"
        elif probability < 0.6:
            confidence = "Riesgo moderado de Churn"
        else:
            confidence = "Alto riesgo de Churn"
        
        return ChurnPredictionResponse(
            prediction="Yes" if prediction == 1 else "No",
            probability=float(probability),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@ml_router.post("/predict/knn", response_model=ChurnPredictionResponse)
async def predict_churn_knn(customer: TelcoCustomerInput):
    """
    Predice Churn usando K-Nearest Neighbors
    """
    try:
        # Convertir input a diccionario
        customer_dict = customer.model_dump()
        
        # Codificar variables categóricas
        for col, encoder in label_encoders.items():
            if col in customer_dict:
                try:
                    customer_dict[col] = encoder.transform([customer_dict[col]])[0]
                except ValueError:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Valor inválido para {col}: {customer_dict[col]}"
                    )
        
        # Crear array con el orden correcto de features
        features = np.array([customer_dict[col] for col in feature_names]).reshape(1, -1)
        
        # Normalizar
        features_scaled = knn_scaler.transform(features)
        
        # Predecir
        prediction = knn_model.predict(features_scaled)[0]
        probability = knn_model.predict_proba(features_scaled)[0][1]
        
        # Determinar nivel de confianza
        if probability < 0.3:
            confidence = "Bajo riesgo de Churn"
        elif probability < 0.6:
            confidence = "Riesgo moderado de Churn"
        else:
            confidence = "Alto riesgo de Churn"
        
        return ChurnPredictionResponse(
            prediction="Yes" if prediction == 1 else "No",
            probability=float(probability),
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@ml_router.post("/predict/kmeans", response_model=ClusterPredictionResponse)
async def predict_cluster_kmeans(card: CreditCardInput):
    """
    Predice el cluster usando K-Means
    """
    try:
        # Convertir input a diccionario
        card_dict = card.model_dump()
        
        # Crear array con el orden correcto de features
        features = np.array([card_dict[col] for col in kmeans_feature_names]).reshape(1, -1)
        
        # Normalizar
        features_scaled = kmeans_scaler.transform(features)
        
        # Predecir cluster
        cluster = int(kmeans_model.predict(features_scaled)[0])
        
        # Obtener descripción del cluster
        cluster_info = cluster_descriptions[str(cluster)]
        
        return ClusterPredictionResponse(
            cluster=cluster,
            cluster_name=cluster_info['name'],
            cluster_description=cluster_info['description'],
            avg_balance=cluster_info['avg_balance'],
            avg_purchases=cluster_info['avg_purchases'],
            avg_credit_limit=cluster_info['avg_credit_limit'],
            customer_count=cluster_info['customer_count']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción de cluster: {str(e)}")

@ml_router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Obtiene las métricas de todos los modelos
    """
    return MetricsResponse(
        logistic_regression=logistic_metrics,
        knn=knn_metrics,
        kmeans=kmeans_metrics
    )

@ml_router.get("/clusters/descriptions")
async def get_cluster_descriptions():
    """
    Obtiene las descripciones de todos los clusters
    """
    return cluster_descriptions
