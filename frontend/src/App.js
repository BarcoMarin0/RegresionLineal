import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './App.css';

// Páginas
import Home from './pages/Home';
import LogisticRegression from './pages/LogisticRegression';
import KNN from './pages/KNN';
import KMeans from './pages/KMeans';
import Comparacion from './pages/Comparacion';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/logistic" element={<LogisticRegression />} />
          <Route path="/knn" element={<KNN />} />
          <Route path="/kmeans" element={<KMeans />} />
          <Route path="/comparacion" element={<Comparacion />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
