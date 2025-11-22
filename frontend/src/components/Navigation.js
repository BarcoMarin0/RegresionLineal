import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Home, ArrowLeft } from 'lucide-react';

const Navigation = ({ title }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <nav style={styles.nav}>
      <div style={styles.container}>
        <button
          onClick={() => navigate('/')}
          style={styles.backBtn}
          data-testid="nav-home-btn"
        >
          {isHome ? <Home size={20} /> : <ArrowLeft size={20} />}
          <span style={styles.btnText}>{isHome ? 'Inicio' : 'Volver'}</span>
        </button>
        <h2 style={styles.title}>{title}</h2>
        <div style={{ width: '100px' }}></div>
      </div>
    </nav>
  );
};

const styles = {
  nav: {
    background: 'rgba(255, 255, 255, 0.03)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
    padding: '1rem 0',
    position: 'sticky',
    top: 0,
    zIndex: 100,
    backdropFilter: 'blur(10px)'
  },
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 2rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  backBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    background: 'rgba(255, 255, 255, 0.05)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '8px',
    padding: '0.5rem 1rem',
    color: '#e8eaed',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    fontSize: '0.9rem',
    fontWeight: '500'
  },
  btnText: {
    fontFamily: 'Space Grotesk, sans-serif'
  },
  title: {
    color: '#e8eaed',
    fontSize: '1.25rem',
    fontWeight: '600',
    fontFamily: 'Space Grotesk, sans-serif'
  }
};

export default Navigation;
