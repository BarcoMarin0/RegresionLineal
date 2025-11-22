import React from 'react';
import { useNavigate } from 'react-router-dom';
import { BarChart3, Brain, GitBranch, TrendingUp } from 'lucide-react';

const Home = () => {
  const navigate = useNavigate();

  const models = [
    {
      id: 'logistic',
      title: 'Regresión Logística',
      description: 'Predice la probabilidad de Churn de clientes de telecomunicaciones',
      icon: <TrendingUp size={32} />,
      color: '#4f9cf9',
      path: '/logistic'
    },
    {
      id: 'knn',
      title: 'K-Nearest Neighbors',
      description: 'Clasifica clientes según sus vecinos más cercanos para predecir Churn',
      icon: <GitBranch size={32} />,
      color: '#06a77d',
      path: '/knn'
    },
    {
      id: 'kmeans',
      title: 'K-Means Clustering',
      description: 'Segmenta clientes de tarjetas de crédito en grupos con comportamiento similar',
      icon: <Brain size={32} />,
      color: '#f59e0b',
      path: '/kmeans'
    },
    {
      id: 'comparacion',
      title: 'Comparación de Modelos',
      description: 'Compara métricas y rendimiento de los modelos supervisados',
      icon: <BarChart3 size={32} />,
      color: '#8b5cf6',
      path: '/comparacion'
    }
  ];

  return (
    <div className="home-container">
      <style>{`
        .home-container {
          min-height: 100vh;
          padding: 3rem 2rem;
          background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        }

        .home-header {
          text-align: center;
          margin-bottom: 4rem;
          animation: fadeInDown 0.8s ease;
        }

        .home-title {
          font-size: 3.5rem;
          font-weight: 700;
          margin-bottom: 1rem;
          background: linear-gradient(135deg, #4f9cf9 0%, #6fb1fc 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .home-subtitle {
          font-size: 1.25rem;
          color: #9ca3af;
          max-width: 700px;
          margin: 0 auto;
          line-height: 1.7;
        }

        .models-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 2rem;
          max-width: 1200px;
          margin: 0 auto;
        }

        .model-card {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 20px;
          padding: 2.5rem;
          cursor: pointer;
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
          animation: fadeInUp 0.8s ease backwards;
        }

        .model-card:nth-child(1) { animation-delay: 0.1s; }
        .model-card:nth-child(2) { animation-delay: 0.2s; }
        .model-card:nth-child(3) { animation-delay: 0.3s; }
        .model-card:nth-child(4) { animation-delay: 0.4s; }

        .model-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 3px;
          background: var(--card-color);
          transform: scaleX(0);
          transition: transform 0.4s ease;
        }

        .model-card:hover::before {
          transform: scaleX(1);
        }

        .model-card:hover {
          transform: translateY(-8px);
          border-color: var(--card-color);
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4),
                      0 0 0 1px var(--card-color);
        }

        .model-icon {
          width: 64px;
          height: 64px;
          border-radius: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 1.5rem;
          background: var(--card-color-alpha);
          color: var(--card-color);
          transition: all 0.3s ease;
        }

        .model-card:hover .model-icon {
          transform: scale(1.1) rotate(5deg);
        }

        .model-title {
          font-size: 1.5rem;
          font-weight: 600;
          margin-bottom: 0.75rem;
          color: #e8eaed;
        }

        .model-description {
          font-size: 0.95rem;
          color: #9ca3af;
          line-height: 1.6;
        }

        .footer {
          text-align: center;
          margin-top: 5rem;
          padding-top: 2rem;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          color: #6b7280;
        }

        @keyframes fadeInDown {
          from {
            opacity: 0;
            transform: translateY(-30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @media (max-width: 768px) {
          .home-title {
            font-size: 2.5rem;
          }
          
          .home-subtitle {
            font-size: 1rem;
          }

          .models-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
          }

          .model-card {
            padding: 2rem;
          }
        }
      `}</style>

      <div className="home-header">
        <h1 className="home-title" data-testid="home-title">
          Machine Learning Aplicado
        </h1>
        <p className="home-subtitle" data-testid="home-subtitle">
          Proyecto integrador de modelos supervisados y no supervisados con despliegue web.
          Explora y prueba diferentes algoritmos de Machine Learning en tiempo real.
        </p>
      </div>

      <div className="models-grid">
        {models.map((model) => (
          <div
            key={model.id}
            className="model-card"
            onClick={() => navigate(model.path)}
            style={{
              '--card-color': model.color,
              '--card-color-alpha': `${model.color}20`
            }}
            data-testid={`model-card-${model.id}`}
          >
            <div className="model-icon">
              {model.icon}
            </div>
            <h3 className="model-title">{model.title}</h3>
            <p className="model-description">{model.description}</p>
          </div>
        ))}
      </div>

      <div className="footer">
        <p>Proyecto Final - Machine Learning Aplicado 2025</p>
      </div>
    </div>
  );
};

export default Home;
