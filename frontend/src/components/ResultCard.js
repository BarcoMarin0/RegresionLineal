import React from 'react';
import { CheckCircle, AlertCircle, Info } from 'lucide-react';

const ResultCard = ({ result, type = 'churn' }) => {
  if (!result) return null;

  const getChurnIcon = () => {
    if (result.prediction === 'Yes') {
      return <AlertCircle size={32} color="#ef4444" />;
    }
    return <CheckCircle size={32} color="#10b981" />;
  };

  const getConfidenceColor = () => {
    if (result.probability < 0.3) return '#10b981';
    if (result.probability < 0.6) return '#f59e0b';
    return '#ef4444';
  };

  if (type === 'cluster') {
    return (
      <div style={styles.card} data-testid="result-card">
        <div style={styles.header}>
          <Info size={32} color="#4f9cf9" />
          <h3 style={styles.title}>Resultado del Clustering</h3>
        </div>
        
        <div style={styles.content}>
          <div style={styles.resultItem}>
            <span style={styles.label}>Cluster Asignado:</span>
            <span style={{ ...styles.value, color: '#4f9cf9' }}>
              Cluster {result.cluster}
            </span>
          </div>
          
          <div style={styles.resultItem}>
            <span style={styles.label}>Nombre del Segmento:</span>
            <span style={styles.value}>{result.cluster_name}</span>
          </div>
          
          <div style={{ ...styles.descriptionBox, borderColor: '#4f9cf940' }}>
            <p style={styles.description}>{result.cluster_description}</p>
          </div>
          
          <div style={styles.statsGrid}>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Balance Promedio</span>
              <span style={styles.statValue}>${result.avg_balance.toLocaleString('es-ES', { maximumFractionDigits: 2 })}</span>
            </div>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Compras Promedio</span>
              <span style={styles.statValue}>${result.avg_purchases.toLocaleString('es-ES', { maximumFractionDigits: 2 })}</span>
            </div>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Límite de Crédito Promedio</span>
              <span style={styles.statValue}>${result.avg_credit_limit.toLocaleString('es-ES', { maximumFractionDigits: 2 })}</span>
            </div>
            <div style={styles.stat}>
              <span style={styles.statLabel}>Clientes en el Cluster</span>
              <span style={styles.statValue}>{result.customer_count.toLocaleString('es-ES')}</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.card} data-testid="result-card">
      <div style={styles.header}>
        {getChurnIcon()}
        <h3 style={styles.title}>Resultado de la Predicción</h3>
      </div>
      
      <div style={styles.content}>
        <div style={styles.resultItem}>
          <span style={styles.label}>Predicción de Churn:</span>
          <span style={{
            ...styles.value,
            color: result.prediction === 'Yes' ? '#ef4444' : '#10b981'
          }}>
            {result.prediction === 'Yes' ? 'Sí (Cliente en Riesgo)' : 'No (Cliente Estable)'}
          </span>
        </div>
        
        <div style={styles.resultItem}>
          <span style={styles.label}>Probabilidad de Churn:</span>
          <span style={{ ...styles.value, color: getConfidenceColor() }}>
            {(result.probability * 100).toFixed(2)}%
          </span>
        </div>
        
        <div style={styles.progressBar}>
          <div
            style={{
              ...styles.progressFill,
              width: `${result.probability * 100}%`,
              background: getConfidenceColor()
            }}
          />
        </div>
        
        <div style={{ ...styles.descriptionBox, borderColor: `${getConfidenceColor()}40` }}>
          <p style={styles.description}>{result.confidence}</p>
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
    marginTop: '2rem',
    animation: 'fadeIn 0.5s ease'
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1.5rem',
    paddingBottom: '1rem',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
  },
  title: {
    color: '#e8eaed',
    fontSize: '1.5rem',
    fontWeight: '600',
    fontFamily: 'Space Grotesk, sans-serif'
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem'
  },
  resultItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '1rem',
    background: 'rgba(255, 255, 255, 0.02)',
    borderRadius: '8px'
  },
  label: {
    color: '#9ca3af',
    fontSize: '0.95rem'
  },
  value: {
    fontSize: '1.1rem',
    fontWeight: '600',
    fontFamily: 'Space Grotesk, sans-serif'
  },
  progressBar: {
    width: '100%',
    height: '8px',
    background: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '4px',
    overflow: 'hidden'
  },
  progressFill: {
    height: '100%',
    transition: 'width 0.6s ease'
  },
  descriptionBox: {
    padding: '1rem',
    background: 'rgba(255, 255, 255, 0.02)',
    border: '1px solid',
    borderRadius: '8px'
  },
  description: {
    color: '#c5c9d1',
    fontSize: '0.95rem',
    lineHeight: '1.6',
    margin: 0
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '1rem'
  },
  stat: {
    padding: '1rem',
    background: 'rgba(255, 255, 255, 0.02)',
    borderRadius: '8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem'
  },
  statLabel: {
    color: '#9ca3af',
    fontSize: '0.85rem'
  },
  statValue: {
    color: '#e8eaed',
    fontSize: '1.1rem',
    fontWeight: '600',
    fontFamily: 'Space Grotesk, sans-serif'
  }
};

export default ResultCard;
