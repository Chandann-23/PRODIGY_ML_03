import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LayoutDashboard, Database, Cpu, Award, Eye, BrainCircuit } from 'lucide-react';
import Dashboard from './components/Dashboard';
import DatasetManager from './components/DatasetManager';
import PCABoundaryCanvas from './components/PCABoundaryCanvas';
import HyperparameterSweep from './components/HyperparameterSweep';
import ModelRegistry from './components/ModelRegistry';
import PredictorPanel from './components/PredictorPanel';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [samples, setSamples] = useState([]);
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const [projectedSamples, setProjectedSamples] = useState([]);
  const [testPoint, setTestPoint] = useState(null);

  const fetchData = async () => {
    try {
      // Fetch samples list (for count and metadata)
      const samplesRes = await axios.get('/api/samples');
      setSamples(samplesRes.data);

      // Fetch active model
      try {
        const activeModelRes = await axios.get('/api/model/active');
        setActiveModel(activeModelRes.data);
      } catch (err) {
        if (err.response?.status === 404) {
          setActiveModel(null);
        } else {
          console.error('Error fetching active model:', err);
        }
      }

      // Fetch all models for registry
      const modelsRes = await axios.get('/api/models');
      setModels(modelsRes.data);

      // Fetch 2D projected samples (if active model exists)
      const projectedRes = await axios.get('/api/samples/projected');
      setProjectedSamples(projectedRes.data);

    } catch (err) {
      console.error('Failed to load application data:', err);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleModelTrained = (newModel, newProjectedSamples) => {
    setActiveModel(newModel);
    setProjectedSamples(newProjectedSamples);
    setTestPoint(null); // Clear old test points on new training run
    
    // Refresh registry history
    axios.get('/api/models').then(res => setModels(res.data));
  };

  const handlePredictionResult = (point) => {
    setTestPoint(point);
  };

  const clearTestPoint = () => {
    setTestPoint(null);
  };

  // Determine whether to show the PCA sidebar
  const showSidebar = activeModel && (activeTab === 'dashboard' || activeTab === 'sweep' || activeTab === 'predict');

  return (
    <div className="app-container">
      
      {/* Header bar */}
      <header className="header">
        <div className="title-group">
          <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <BrainCircuit size={32} color="var(--primary-glow)" style={{ filter: 'drop-shadow(0 0 10px var(--primary-glow))' }} />
            Hyperplane.ai
          </h1>
          <p>Interactive High-Dimensional SVM Engine & MLOps Pipeline</p>
        </div>

        {/* Tab Controls */}
        <nav className="nav-tabs">
          <button 
            className={`nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <LayoutDashboard size={16} />
            Dashboard
          </button>
          <button 
            className={`nav-tab ${activeTab === 'dataset' ? 'active' : ''}`}
            onClick={() => setActiveTab('dataset')}
          >
            <Database size={16} />
            Dataset Manager
          </button>
          <button 
            className={`nav-tab ${activeTab === 'sweep' ? 'active' : ''}`}
            onClick={() => setActiveTab('sweep')}
          >
            <Cpu size={16} />
            Hyperparameter Sweep
          </button>
          <button 
            className={`nav-tab ${activeTab === 'registry' ? 'active' : ''}`}
            onClick={() => setActiveTab('registry')}
          >
            <Award size={16} />
            Model Registry
          </button>
          <button 
            className={`nav-tab ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => setActiveTab('predict')}
          >
            <Eye size={16} />
            Production Predictor
          </button>
        </nav>
      </header>

      {/* Main Content Layout */}
      <main className={`main-layout ${showSidebar ? 'has-sidebar' : ''}`}>
        
        {/* Active tab content panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {activeTab === 'dashboard' && (
            <Dashboard 
              activeModel={activeModel} 
              samples={samples} 
              onModelTrained={handleModelTrained}
              onTabChange={setActiveTab}
            />
          )}

          {activeTab === 'dataset' && (
            <DatasetManager 
              samples={samples} 
              setSamples={setSamples}
              onDatasetChange={fetchData}
            />
          )}

          {activeTab === 'sweep' && (
            <HyperparameterSweep 
              samples={samples}
            />
          )}

          {activeTab === 'registry' && (
            <ModelRegistry 
              models={models}
              activeModel={activeModel}
              onModelDeployed={fetchData}
              onModelDeleted={fetchData}
              refreshRegistry={fetchData}
            />
          )}

          {activeTab === 'predict' && (
            <PredictorPanel 
              activeModel={activeModel}
              onPredictionResult={handlePredictionResult}
              clearTestPoint={clearTestPoint}
            />
          )}
        </div>

        {/* Sidebar PCA Decision boundary visualization */}
        {showSidebar && (
          <div style={{ position: 'sticky', top: '2rem' }}>
            <PCABoundaryCanvas 
              model={activeModel}
              samples={projectedSamples}
              testPoint={testPoint}
            />
          </div>
        )}

      </main>
    </div>
  );
}
