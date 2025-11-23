import React, { useState } from 'react';
import axios from 'axios';
import Navigation from '../components/Navigation';
import ResultCard from '../components/ResultCard';
import { Loader2 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const LogisticRegression = () => {
  const [formData, setFormData] = useState({
    gender: 'Male',
    SeniorCitizen: 0,
    Partner: 'No',
    Dependents: 'No',
    tenure: 12,
    PhoneService: 'Yes',
    MultipleLines: 'No',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'No',
    StreamingMovies: 'No',
    Contract: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 70.0,
    TotalCharges: 840.0
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/ml/predict/logistic`,
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
      <Navigation title="Regresión Logística" />
      
      <div style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: '700', color: '#e8eaed', marginBottom: '0.5rem' }}>
            Predicción de Churn con <span style={{ color: '#4f9cf9' }}>Regresión Logística</span>
          </h1>
          <p style={{ color: '#9ca3af', fontSize: '1.1rem', lineHeight: '1.6' }}>
            Ingresa los datos del cliente para predecir la probabilidad de que abandone el servicio.
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ marginBottom: '2rem' }}>
          <div style={styles.card}>
            <h3 style={styles.sectionTitle}>Información Personal</h3>
            <div style={styles.grid}>
              <div className="form-group">
                <label className="form-label">Género</label>
                <select name="gender" value={formData.gender} onChange={handleChange} className="form-select" data-testid="input-gender">
                  <option value="Male">Masculino</option>
                  <option value="Female">Femenino</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Adulto Mayor</label>
                <select name="SeniorCitizen" value={formData.SeniorCitizen} onChange={handleChange} className="form-select" data-testid="input-senior">
                  <option value={0}>No</option>
                  <option value={1}>Sí</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Tiene Pareja</label>
                <select name="Partner" value={formData.Partner} onChange={handleChange} className="form-select" data-testid="input-partner">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Tiene Dependientes</label>
                <select name="Dependents" value={formData.Dependents} onChange={handleChange} className="form-select" data-testid="input-dependents">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Antigüedad (meses)</label>
                <input
                  type="number"
                  name="tenure"
                  value={formData.tenure}
                  onChange={handleChange}
                  min="0"
                  max="72"
                  className="form-input"
                  data-testid="input-tenure"
                />
              </div>
            </div>
          </div>

          <div style={styles.card}>
            <h3 style={styles.sectionTitle}>Servicios</h3>
            <div style={styles.grid}>
              <div className="form-group">
                <label className="form-label">Servicio Telefónico</label>
                <select name="PhoneService" value={formData.PhoneService} onChange={handleChange} className="form-select" data-testid="input-phone">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Múltiples Líneas</label>
                <select name="MultipleLines" value={formData.MultipleLines} onChange={handleChange} className="form-select" data-testid="input-multiple-lines">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                  <option value="No phone service">Sin servicio telefónico</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Servicio de Internet</label>
                <select name="InternetService" value={formData.InternetService} onChange={handleChange} className="form-select" data-testid="input-internet">
                  <option value="DSL">DSL</option>
                  <option value="Fiber optic">Fibra óptica</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Seguridad Online</label>
                <select name="OnlineSecurity" value={formData.OnlineSecurity} onChange={handleChange} className="form-select" data-testid="input-security">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                  <option value="No internet service">Sin servicio de internet</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Respaldo Online</label>
                <select name="OnlineBackup" value={formData.OnlineBackup} onChange={handleChange} className="form-select" data-testid="input-backup">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                  <option value="No internet service">Sin servicio de internet</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Protección de Dispositivo</label>
                <select name="DeviceProtection" value={formData.DeviceProtection} onChange={handleChange} className="form-select" data-testid="input-device">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                  <option value="No internet service">Sin servicio de internet</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Soporte Técnico</label>
                <select name="TechSupport" value={formData.TechSupport} onChange={handleChange} className="form-select" data-testid="input-support">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                  <option value="No internet service">Sin servicio de internet</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Streaming TV</label>
                <select name="StreamingTV" value={formData.StreamingTV} onChange={handleChange} className="form-select" data-testid="input-tv">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                  <option value="No internet service">Sin servicio de internet</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Streaming Películas</label>
                <select name="StreamingMovies" value={formData.StreamingMovies} onChange={handleChange} className="form-select" data-testid="input-movies">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                  <option value="No internet service">Sin servicio de internet</option>
                </select>
              </div>
            </div>
          </div>

          <div style={styles.card}>
            <h3 style={styles.sectionTitle}>Facturación y Pagos</h3>
            <div style={styles.grid}>
              <div className="form-group">
                <label className="form-label">Tipo de Contrato</label>
                <select name="Contract" value={formData.Contract} onChange={handleChange} className="form-select" data-testid="input-contract">
                  <option value="Month-to-month">Mes a mes</option>
                  <option value="One year">Un año</option>
                  <option value="Two year">Dos años</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Factura Electrónica</label>
                <select name="PaperlessBilling" value={formData.PaperlessBilling} onChange={handleChange} className="form-select" data-testid="input-paperless">
                  <option value="Yes">Sí</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Método de Pago</label>
                <select name="PaymentMethod" value={formData.PaymentMethod} onChange={handleChange} className="form-select" data-testid="input-payment">
                  <option value="Electronic check">Cheque electrónico</option>
                  <option value="Mailed check">Cheque por correo</option>
                  <option value="Bank transfer (automatic)">Transferencia bancaria (automática)</option>
                  <option value="Credit card (automatic)">Tarjeta de crédito (automática)</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Cargos Mensuales ($)</label>
                <input
                  type="number"
                  name="MonthlyCharges"
                  value={formData.MonthlyCharges}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-monthly"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Cargos Totales ($)</label>
                <input
                  type="number"
                  name="TotalCharges"
                  value={formData.TotalCharges}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  className="form-input"
                  data-testid="input-total"
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: '100%', marginTop: '1rem' }}
            data-testid="submit-button"
          >
            {loading ? (
              <>
                <Loader2 size={20} style={{ animation: 'spin 1s linear infinite', marginRight: '0.5rem' }} />
                Procesando...
              </>
            ) : (
              'Predecir Churn'
            )}
          </button>
        </form>

        {error && (
          <div style={{ ...styles.card, borderColor: '#ef4444', background: 'rgba(239, 68, 68, 0.1)' }}>
            <p style={{ color: '#ef4444', margin: 0 }}>{error}</p>
          </div>
        )}

        {result && <ResultCard result={result} type="churn" />}
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

export default LogisticRegression;
