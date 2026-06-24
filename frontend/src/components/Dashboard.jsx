import React, { useState } from 'react';
import axios from 'axios';
import confetti from 'canvas-confetti';
import { Sliders, Cpu, Activity, Play, CheckCircle, Database } from 'lucide-react';

export default function Dashboard({ activeModel, samples, onModelTrained, onTabChange }) {
  const [kernel, setKernel] = useState('linear');
  const [C, setC] = useState(1.0);
  const [gamma, setGamma] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleTrain = async () => {
    if (samples.length < 4) return;
    setLoading(true);
    setError('');
    try {
      const res = await axios.post('/api/train', {
        kernel,
        C: parseFloat(C),
        gamma: parseFloat(gamma)
      });
      
      onModelTrained(res.data.model, res.data.projectedSamples);

      // Trigger Confetti Celebration!
      confetti({
        particleCount: 150,
        spread: 80,
        origin: { y: 0.6 },
        colors: ['#6366f1', '#06b6d4', '#ec4899', '#10b981']
      });

    } catch (err) {
      setError(err.response?.data?.error || 'Failed to train: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const dogCount = samples.filter(s => s.label === 1).length;
  const catCount = samples.filter(s => s.label === -1).length;

  return (
    <div className="dashboard-grid">
      
      {/* Configuration & Control Panel */}
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sliders size={20} color="var(--primary-glow)" />
          SVM Hyperparameter Configurator
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
          Adjust the Support Vector Machine constraints. Training utilizes the Sequential Minimal Optimization (SMO) solver in Node.js.
        </p>

        {error && (
          <div className="alert alert-danger" style={{ background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)', border: '1px solid rgba(239, 68, 68, 0.25)' }}>
            <div>{error}</div>
          </div>
        )}

        {/* Kernel Selection */}
        <div className="form-group">
          <label className="form-label">Kernel Function</label>
          <select 
            value={kernel} 
            onChange={(e) => setKernel(e.target.value)}
            disabled={loading}
            className="form-select"
          >
            <option value="linear">Linear Kernel: K(x, y) = x · y</option>
            <option value="rbf">Radial Basis Function (RBF) Kernel: K(x, y) = exp(-γ||x - y||²)</option>
          </select>
        </div>

        {/* Box Constraint C */}
        <div className="form-group">
          <label className="form-label">Box Constraint C (Regularization Strength)</label>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Controls trade-off between maximizing margin and minimizing training errors. High C tries to classify all training examples correctly.
          </p>
          <div className="range-group">
            <input
              type="range"
              min="0.1"
              max="15.0"
              step="0.1"
              value={C}
              onChange={(e) => setC(e.target.value)}
              disabled={loading}
              className="range-input"
            />
            <span className="range-val">{C}</span>
          </div>
        </div>

        {/* Gamma Parameter for RBF */}
        {kernel === 'rbf' && (
          <div className="form-group" style={{ animation: 'fadeIn 0.3s' }}>
            <label className="form-label">RBF Gamma (γ)</label>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
              Defines how far the influence of a single training example reaches. Low values mean 'far', high values mean 'close'.
            </p>
            <div className="range-group">
              <input
                type="range"
                min="0.01"
                max="2.0"
                step="0.01"
                value={gamma}
                onChange={(e) => setGamma(e.target.value)}
                disabled={loading}
                className="range-input"
              />
              <span className="range-val">{gamma}</span>
            </div>
          </div>
        )}

        {/* Train Button or CTA */}
        {samples.length < 4 ? (
          <div style={{
            background: 'rgba(99, 102, 241, 0.05)',
            border: '1px dashed var(--border-color)',
            borderRadius: '10px',
            padding: '1rem',
            textAlign: 'center',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            marginTop: '1rem'
          }}>
            <Database size={24} style={{ margin: '0 auto 0.5rem', opacity: 0.7 }} />
            <p>Dataset size is too small to train.</p>
            <button 
              className="btn btn-primary" 
              onClick={() => onTabChange('dataset')}
              style={{ marginTop: '0.75rem', padding: '0.4rem 1rem', fontSize: '0.8rem' }}
            >
              Go to Dataset Manager
            </button>
          </div>
        ) : (
          <button
            onClick={handleTrain}
            disabled={loading}
            className="btn btn-primary"
            style={{ marginTop: '1rem', width: '100%' }}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                <span>Solving SMO Quadratic Programming...</span>
              </>
            ) : (
              <>
                <Play size={18} fill="#fff" />
                <span>Train Support Vector Machine (SVM)</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Model Performance metrics Dashboard */}
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--primary-glow)" />
            Active Model Analytics
          </h3>

          {!activeModel ? (
            <div style={{
              flexGrow: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-muted)',
              padding: '4rem 1rem',
              textAlign: 'center',
              border: '1px dashed var(--border-color)',
              borderRadius: '12px',
              background: 'rgba(0,0,0,0.1)'
            }}>
              <Cpu size={40} style={{ marginBottom: '1rem', opacity: 0.5 }} />
              <p style={{ fontWeight: 600 }}>No Active Classifier Deployed</p>
              <p style={{ fontSize: '0.8rem', marginTop: '0.2rem', maxWidth: '280px' }}>
                Tune parameters and click "Train Support Vector Machine" to deploy your first model run to MongoDB Atlas.
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', animation: 'fadeIn 0.5s' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                <div>
                  <h4 style={{ fontWeight: 700, fontSize: '1.1rem', color: '#fff' }}>{activeModel.name}</h4>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Model Registry Version: {activeModel.version}</span>
                </div>
                <span className="badge badge-active">Active</span>
              </div>

              {/* Grid Metrics */}
              <div className="grid-cols-4" style={{ gap: '0.75rem', marginBottom: 0 }}>
                <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '0.75rem', borderRadius: '10px', border: '1px solid var(--border-color)', textAlign: 'center' }}>
                  <div className="metric-val" style={{ color: 'var(--success)' }}>
                    {Math.round(activeModel.metrics.accuracy * 100)}%
                  </div>
                  <div className="metric-label" style={{ fontSize: '0.65rem' }}>Validation Acc</div>
                </div>

                <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '0.75rem', borderRadius: '10px', border: '1px solid var(--border-color)', textAlign: 'center' }}>
                  <div className="metric-val" style={{ color: 'var(--primary-glow)' }}>
                    {activeModel.metrics.f1.toFixed(3)}
                  </div>
                  <div className="metric-label" style={{ fontSize: '0.65rem' }}>F1-Score</div>
                </div>

                <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '0.75rem', borderRadius: '10px', border: '1px solid var(--border-color)', textAlign: 'center' }}>
                  <div className="metric-val" style={{ color: '#fbbf24' }}>
                    {activeModel.supportVectorCount}
                  </div>
                  <div className="metric-label" style={{ fontSize: '0.65rem' }}>Support Vectors</div>
                </div>

                <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '0.75rem', borderRadius: '10px', border: '1px solid var(--border-color)', textAlign: 'center' }}>
                  <div className="metric-val" style={{ color: 'var(--text-primary)' }}>
                    {activeModel.metrics.trainingTimeMs}ms
                  </div>
                  <div className="metric-label" style={{ fontSize: '0.65rem' }}>Training Time</div>
                </div>
              </div>

              {/* Confusion Matrix visual */}
              <div>
                <h5 style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.5rem', textAlign: 'center' }}>
                  Validation Split Confusion Matrix
                </h5>
                <div className="matrix-container">
                  <div></div>
                  <div className="matrix-header-cell">Pred Cat</div>
                  <div className="matrix-header-cell">Pred Dog</div>
                  <div className="matrix-label-cell">True Cat</div>
                  <div className="matrix-cell tn">
                    <div className="val">{activeModel.metrics.confusionMatrix.tn}</div>
                    <div style={{ fontSize: '0.55rem' }}>TN</div>
                  </div>
                  <div className="matrix-cell">
                    <div className="val">{activeModel.metrics.confusionMatrix.fp}</div>
                    <div style={{ fontSize: '0.55rem' }}>FP</div>
                  </div>
                  <div className="matrix-label-cell">True Dog</div>
                  <div className="matrix-cell">
                    <div className="val">{activeModel.metrics.confusionMatrix.fn}</div>
                    <div style={{ fontSize: '0.55rem' }}>FN</div>
                  </div>
                  <div className="matrix-cell tp">
                    <div className="val">{activeModel.metrics.confusionMatrix.tp}</div>
                    <div style={{ fontSize: '0.55rem' }}>TP</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {activeModel && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '8px', padding: '0.75rem', fontSize: '0.8rem', color: 'var(--success)' }}>
            <CheckCircle size={16} />
            <span>Active model deployed to production endpoint. Ready for inference predictions.</span>
          </div>
        )}
      </div>

    </div>
  );
}
