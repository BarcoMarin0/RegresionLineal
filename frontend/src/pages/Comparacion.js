import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navigation from '../components/Navigation';
import { Loader2, TrendingUp, Award } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const Comparacion = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/ml/metrics`);
      setMetrics(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error al cargar las métricas');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%)' }}>
        <Navigation title="Comparación de Modelos" />
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 'calc(100vh - 100px)' }}>
          <Loader2 size={48} style={{ animation: 'spin 1s linear infinite', color: '#4f9cf9' }} />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%)' }}>
        <Navigation title="Comparación de Modelos" />
        <div style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem' }}>
          <div style={{ ...styles.card, borderColor: '#ef4444', background: 'rgba(239, 68, 68, 0.1)' }}>
            <p style={{ color: '#ef4444', margin: 0 }}>{error}</p>
          </div>
        </div>
      </div>
    );
  }

  const logistic = metrics?.logistic_regression;
  const knn = metrics?.knn;

  // Determinar el mejor modelo
  const getBestModel = (metric) => {
    if (!logistic || !knn) return null;
    return logistic[metric] > knn[metric] ? 'logistic' : 'knn';
  };

  const MetricComparison = ({ name, logisticValue, knnValue, format = 'percentage' }) => {
    const best = logisticValue > knnValue ? 'logistic' : 'knn';
    
    const formatValue = (val) => {
      if (format === 'percentage') return `${(val * 100).toFixed(2)}%`;
      if (format === 'number') return val.toFixed(4);
      return val;
    };

    return (
      <div style={styles.metricRow}>
        <div style={styles.metricName}>{name}</div>
        <div style={styles.metricValues}>
          <div style={{
            ...styles.metricValue,
            background: best === 'logistic' ? 'rgba(79, 156, 249, 0.1)' : 'rgba(255, 255, 255, 0.02)',
            border: best === 'logistic' ? '1px solid rgba(79, 156, 249, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)'
          }}>
            <span style={styles.metricLabel}>Regresión Logística</span>
            <span style={styles.metricNumber}>{formatValue(logisticValue)}</span>
            {best === 'logistic' && <Award size={16} color="#4f9cf9" style={{ marginLeft: '0.5rem' }} />}
          </div>
          <div style={{
            ...styles.metricValue,
            background: best === 'knn' ? 'rgba(6, 167, 125, 0.1)' : 'rgba(255, 255, 255, 0.02)',
            border: best === 'knn' ? '1px solid rgba(6, 167, 125, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)'
          }}>
            <span style={styles.metricLabel}>KNN (K={knn?.k})</span>
            <span style={styles.metricNumber}>{formatValue(knnValue)}</span>
            {best === 'knn' && <Award size={16} color="#06a77d" style={{ marginLeft: '0.5rem' }} />}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%)' }}>
      <Navigation title="Comparación de Modelos" />
      
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: '700', color: '#e8eaed', marginBottom: '0.5rem' }}>
            Comparación de <span style={{ color: '#8b5cf6' }}>Modelos Supervisados</span>
          </h1>
          <p style={{ color: '#9ca3af', fontSize: '1.1rem', lineHeight: '1.6' }}>
            Análisis comparativo de métricas entre Regresión Logística y K-Nearest Neighbors para predicción de Churn.
          </p>
        </div>

        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <TrendingUp size={24} color="#8b5cf6" />
            <h3 style={styles.cardTitle}>Métricas de Desempeño</h3>
          </div>

          <div style={styles.metricsContainer}>
            <MetricComparison 
              name="Accuracy (Exactitud)" 
              logisticValue={logistic?.accuracy} 
              knnValue={knn?.accuracy}
            />
            <MetricComparison 
              name="Precision (Precisión)" 
              logisticValue={logistic?.precision} 
              knnValue={knn?.precision}
            />
            <MetricComparison 
              name="Recall (Sensibilidad)" 
              logisticValue={logistic?.recall} 
              knnValue={knn?.recall}
            />
            <MetricComparison 
              name="F1-Score" 
              logisticValue={logistic?.f1_score} 
              knnValue={knn?.f1_score}
            />
            <MetricComparison 
              name="AUC (Área bajo la Curva ROC)" 
              logisticValue={logistic?.auc} 
              knnValue={knn?.auc}
            />
          </div>
        </div>

        <div style={styles.grid}>
          <div style={styles.card}>
            <h3 style={{ ...styles.cardTitle, color: '#4f9cf9', marginBottom: '1.5rem' }}>
              Regresión Logística
            </h3>
            <div style={styles.confusionMatrix}>
              <div style={styles.matrixTitle}>Matriz de Confusión</div>
              <div style={styles.matrixGrid}>
                <div></div>
                <div style={styles.matrixHeader}>Pred: No Churn</div>
                <div style={styles.matrixHeader}>Pred: Churn</div>
                
                <div style={styles.matrixHeader}>Real: No Churn</div>
                <div style={{ ...styles.matrixCell, background: 'rgba(16, 185, 129, 0.1)' }}>
                  {logistic?.confusion_matrix[0][0]}
                </div>
                <div style={{ ...styles.matrixCell, background: 'rgba(239, 68, 68, 0.1)' }}>
                  {logistic?.confusion_matrix[0][1]}
                </div>
                
                <div style={styles.matrixHeader}>Real: Churn</div>
                <div style={{ ...styles.matrixCell, background: 'rgba(239, 68, 68, 0.1)' }}>
                  {logistic?.confusion_matrix[1][0]}
                </div>
                <div style={{ ...styles.matrixCell, background: 'rgba(16, 185, 129, 0.1)' }}>
                  {logistic?.confusion_matrix[1][1]}
                </div>
              </div>
            </div>
            <div style={styles.description}>
              <p>La Regresión Logística proporciona probabilidades interpretables y coeficientes que indican la importancia de cada feature.</p>
            </div>
          </div>

          <div style={styles.card}>
            <h3 style={{ ...styles.cardTitle, color: '#06a77d', marginBottom: '1.5rem' }}>
              K-Nearest Neighbors
            </h3>
            <div style={styles.confusionMatrix}>
              <div style={styles.matrixTitle}>Matriz de Confusión</div>
              <div style={styles.matrixGrid}>
                <div></div>
                <div style={styles.matrixHeader}>Pred: No Churn</div>
                <div style={styles.matrixHeader}>Pred: Churn</div>
                
                <div style={styles.matrixHeader}>Real: No Churn</div>
                <div style={{ ...styles.matrixCell, background: 'rgba(16, 185, 129, 0.1)' }}>
                  {knn?.confusion_matrix[0][0]}
                </div>
                <div style={{ ...styles.matrixCell, background: 'rgba(239, 68, 68, 0.1)' }}>
                  {knn?.confusion_matrix[0][1]}
                </div>
                
                <div style={styles.matrixHeader}>Real: Churn</div>
                <div style={{ ...styles.matrixCell, background: 'rgba(239, 68, 68, 0.1)' }}>
                  {knn?.confusion_matrix[1][0]}
                </div>
                <div style={{ ...styles.matrixCell, background: 'rgba(16, 185, 129, 0.1)' }}>
                  {knn?.confusion_matrix[1][1]}
                </div>
              </div>
            </div>
            <div style={styles.description}>
              <p>KNN con K={knn?.k} clasifica basándose en la similitud con los {knn?.k} vecinos más cercanos en el espacio de características.</p>
            </div>
          </div>
        </div>

        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Conclusiones</h3>
          <div style={styles.conclusions}>
            <div style={styles.conclusionItem}>
              <strong style={{ color: '#4f9cf9' }}>Mejor Accuracy:</strong>
              <span>{getBestModel('accuracy') === 'logistic' ? 'Regresión Logística' : 'KNN'} ({((getBestModel('accuracy') === 'logistic' ? logistic?.accuracy : knn?.accuracy) * 100).toFixed(2)}%)</span>
            </div>
            <div style={styles.conclusionItem}>
              <strong style={{ color: '#4f9cf9' }}>Mejor AUC:</strong>
              <span>{getBestModel('auc') === 'logistic' ? 'Regresión Logística' : 'KNN'} ({(getBestModel('auc') === 'logistic' ? logistic?.auc : knn?.auc).toFixed(4)})</span>
            </div>
            <div style={styles.conclusionItem}>
              <strong style={{ color: '#4f9cf9' }}>Mejor F1-Score:</strong>
              <span>{getBestModel('f1_score') === 'logistic' ? 'Regresión Logística' : 'KNN'} ({((getBestModel('f1_score') === 'logistic' ? logistic?.f1_score : knn?.f1_score) * 100).toFixed(2)}%)</span>
            </div>
          </div>
          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
            <p style={{ color: '#c5c9d1', lineHeight: '1.6', margin: 0 }}>
              {logistic?.accuracy > knn?.accuracy 
                ? 'La Regresión Logística muestra un mejor desempeño general, con mayor precisión y AUC. Es recomendable para escenarios donde se requiere interpretabilidad de los resultados y probabilidades calibradas.'
                : 'KNN muestra un mejor desempeño general en este dataset. Es útil cuando las relaciones entre features no son lineales y se requiere un modelo más flexible.'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles = {
  card: {
    background: 'rgba(255, 255, 255, 0.03)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '16px',
    padding: '2rem',
    marginBottom: '2rem'
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '2rem',
    paddingBottom: '1rem',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
  },
  cardTitle: {
    color: '#e8eaed',
    fontSize: '1.5rem',
    fontWeight: '600',
    fontFamily: 'Space Grotesk, sans-serif',
    margin: 0
  },
  metricsContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem'
  },
  metricRow: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem'
  },
  metricName: {
    color: '#9ca3af',
    fontSize: '0.95rem',
    fontWeight: '500'
  },
  metricValues: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '1rem'
  },
  metricValue: {
    padding: '1rem',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    transition: 'all 0.3s ease'
  },
  metricLabel: {
    color: '#9ca3af',
    fontSize: '0.85rem'
  },
  metricNumber: {
    color: '#e8eaed',
    fontSize: '1.1rem',
    fontWeight: '600',
    fontFamily: 'Space Grotesk, sans-serif'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
    gap: '2rem'
  },
  confusionMatrix: {
    marginBottom: '1.5rem'
  },
  matrixTitle: {
    color: '#9ca3af',
    fontSize: '0.9rem',
    marginBottom: '1rem',
    textAlign: 'center'
  },
  matrixGrid: {
    display: 'grid',
    gridTemplateColumns: '120px 1fr 1fr',
    gap: '0.5rem',
    alignItems: 'center'
  },
  matrixHeader: {
    color: '#9ca3af',
    fontSize: '0.8rem',
    textAlign: 'center',
    padding: '0.5rem'
  },
  matrixCell: {
    padding: '1rem',
    textAlign: 'center',
    borderRadius: '8px',
    color: '#e8eaed',
    fontSize: '1.25rem',
    fontWeight: '600',
    fontFamily: 'Space Grotesk, sans-serif'
  },
  description: {
    padding: '1rem',
    background: 'rgba(255, 255, 255, 0.02)',
    borderRadius: '8px',
    color: '#9ca3af',
    fontSize: '0.9rem',
    lineHeight: '1.6'
  },
  conclusions: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    marginBottom: '1rem'
  },
  conclusionItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '1rem',
    background: 'rgba(255, 255, 255, 0.02)',
    borderRadius: '8px',
    color: '#e8eaed'
  }
};

export default Comparacion;
