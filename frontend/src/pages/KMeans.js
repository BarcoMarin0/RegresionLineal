import React, { useState } from 'react';
import axios from 'axios';
import Navigation from '../components/Navigation';
import ResultCard from '../components/ResultCard';
import { Loader2 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const KMeans = () => {
  const [formData, setFormData] = useState({
    BALANCE: 5000.0,
    PURCHASES: 3000.0,
    ONEOFF_PURCHASES: 1500.0,
    INSTALLMENTS_PURCHASES: 1500.0,
    CASH_ADVANCE: 1000.0,
    PURCHASES_FREQUENCY: 0.5,
    ONEOFF_PURCHASES_FREQUENCY: 0.3,
    PURCHASES_INSTALLMENTS_FREQUENCY: 0.4,
    CASH_ADVANCE_FREQUENCY: 0.2,
    CASH_ADVANCE_TRX: 3,
    PURCHASES_TRX: 15,
    CREDIT_LIMIT: 10000.0,
    PAYMENTS: 4000.0,
    MINIMUM_PAYMENTS: 500.0,
    PRC_FULL_PAYMENT: 0.3
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/ml/predict/kmeans`,
        formData
      );
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error al realizar la predicción');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%)' }}>
      <Navigation title="K-Means Clustering" />
      
      <div style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: '700', color: '#e8eaed', marginBottom: '0.5rem' }}>
            Segmentación con <span style={{ color: '#f59e0b' }}>K-Means</span>
          </h1>
          <p style={{ color: '#9ca3af', fontSize: '1.1rem', lineHeight: '1.6' }}>
            Identifica el segmento al que pertenece un cliente de tarjeta de crédito según su comportamiento financiero.
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ marginBottom: '2rem' }}>
          <div style={styles.card}>
            <h3 style={styles.sectionTitle}>Datos Financieros</h3>
            <div style={styles.grid}>
              <div className="form-group">
                <label className="form-label">Balance ($)</label>
                <input
                  type="number"
                  name="BALANCE"
                  value={formData.BALANCE}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-balance"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Compras Totales ($)</label>
                <input
                  type="number"
                  name="PURCHASES"
                  value={formData.PURCHASES}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-purchases"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Compras Únicas ($)</label>
                <input
                  type="number"
                  name="ONEOFF_PURCHASES"
                  value={formData.ONEOFF_PURCHASES}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-oneoff"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Compras a Plazos ($)</label>
                <input
                  type="number"
                  name="INSTALLMENTS_PURCHASES"
                  value={formData.INSTALLMENTS_PURCHASES}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-installments"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Adelantos en Efectivo ($)</label>
                <input
                  type="number"
                  name="CASH_ADVANCE"
                  value={formData.CASH_ADVANCE}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-cash"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Límite de Crédito ($)</label>
                <input
                  type="number"
                  name="CREDIT_LIMIT"
                  value={formData.CREDIT_LIMIT}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-limit"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Pagos Realizados ($)</label>
                <input
                  type="number"
                  name="PAYMENTS"
                  value={formData.PAYMENTS}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-payments"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Pagos Mínimos ($)</label>
                <input
                  type="number"
                  name="MINIMUM_PAYMENTS"
                  value={formData.MINIMUM_PAYMENTS}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-minimum"
                />
              </div>
            </div>
          </div>

          <div style={styles.card}>
            <h3 style={styles.sectionTitle}>Frecuencias de Uso</h3>
            <div style={styles.grid}>
              <div className="form-group">
                <label className="form-label">Frecuencia de Compras (0-1)</label>
                <input
                  type="number"
                  name="PURCHASES_FREQUENCY"
                  value={formData.PURCHASES_FREQUENCY}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  max="1"
                  className="form-input"
                  data-testid="input-purchase-freq"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Frecuencia de Compras Únicas (0-1)</label>
                <input
                  type="number"
                  name="ONEOFF_PURCHASES_FREQUENCY"
                  value={formData.ONEOFF_PURCHASES_FREQUENCY}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  max="1"
                  className="form-input"
                  data-testid="input-oneoff-freq"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Frecuencia de Compras a Plazos (0-1)</label>
                <input
                  type="number"
                  name="PURCHASES_INSTALLMENTS_FREQUENCY"
                  value={formData.PURCHASES_INSTALLMENTS_FREQUENCY}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  max="1"
                  className="form-input"
                  data-testid="input-installments-freq"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Frecuencia de Adelantos (0-1)</label>
                <input
                  type="number"
                  name="CASH_ADVANCE_FREQUENCY"
                  value={formData.CASH_ADVANCE_FREQUENCY}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  max="1"
                  className="form-input"
                  data-testid="input-cash-freq"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Porcentaje de Pago Completo (0-1)</label>
                <input
                  type="number"
                  name="PRC_FULL_PAYMENT"
                  value={formData.PRC_FULL_PAYMENT}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  max="1"
                  className="form-input"
                  data-testid="input-full-payment"
                />
              </div>
            </div>
          </div>

          <div style={styles.card}>
            <h3 style={styles.sectionTitle}>Transacciones</h3>
            <div style={styles.grid}>
              <div className="form-group">
                <label className="form-label">Número de Transacciones de Compras</label>
                <input
                  type="number"
                  name="PURCHASES_TRX"
                  value={formData.PURCHASES_TRX}
                  onChange={handleChange}
                  min="0"
                  className="form-input"
                  data-testid="input-purchase-trx"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Número de Transacciones de Adelantos</label>
                <input
                  type="number"
                  name="CASH_ADVANCE_TRX"
                  value={formData.CASH_ADVANCE_TRX}
                  onChange={handleChange}
                  min="0"
                  className="form-input"
                  data-testid="input-cash-trx"
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: '100%', marginTop: '1rem', background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)' }}
            data-testid="submit-button"
          >
            {loading ? (
              <>
                <Loader2 size={20} style={{ animation: 'spin 1s linear infinite', marginRight: '0.5rem' }} />
                Procesando...
              </>
            ) : (
              'Identificar Segmento'
            )}
          </button>
        </form>

        {error && (
          <div style={{ ...styles.card, borderColor: '#ef4444', background: 'rgba(239, 68, 68, 0.1)' }}>
            <p style={{ color: '#ef4444', margin: 0 }}>{error}</p>
          </div>
        )}

        {result && <ResultCard result={result} type="cluster" />}
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
  sectionTitle: {
    color: '#e8eaed',
    fontSize: '1.25rem',
    fontWeight: '600',
    marginBottom: '1.5rem',
    fontFamily: 'Space Grotesk, sans-serif'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1.5rem'
  }
};

export default KMeans;
