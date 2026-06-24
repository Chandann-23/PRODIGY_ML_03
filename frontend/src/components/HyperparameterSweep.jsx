import React, { useState } from 'react';
import axios from 'axios';
import { Cpu, Play, Award, CheckCircle, BarChart3, AlertCircle } from 'lucide-react';

export default function HyperparameterSweep({ samples }) {
  const [kernel, setKernel] = useState('rbf');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  // Default grid lists
  const cValues = [0.1, 1.0, 5.0, 10.0];
  const gammaValues = [0.01, 0.1, 0.5, 1.0];

  const handleSweep = async () => {
    if (samples.length < 5) return;
    setLoading(true);
    setError('');
    setResults(null);

    try {
      const res = await axios.post('/api/train/sweep', {
        kernel,
        CValues: cValues,
        gammaValues: kernel === 'rbf' ? gammaValues : [0]
      });

      setResults(res.data.results);
    } catch (err) {
      setError(err.response?.data?.error || 'Sweep failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getHeatmapColor = (acc) => {
    if (acc >= 0.85) return 'rgba(16, 185, 129, 0.4)';  // Glow green
    if (acc >= 0.70) return 'rgba(99, 102, 241, 0.3)';  // Glow indigo
    if (acc >= 0.50) return 'rgba(245, 158, 11, 0.2)';  // Glow yellow
    return 'rgba(239, 68, 68, 0.15)';                  // Fade red
  };

  const getBestConfig = () => {
    if (!results || results.length === 0) return null;
    return results.reduce((best, curr) => curr.accuracy > best.accuracy ? curr : best, results[0]);
  };

  const bestConfig = getBestConfig();

  return (
    <div className="sweep-grid">
      
      {/* Search Config Block */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Cpu size={20} color="var(--primary-glow)" />
          Grid Search Hyperparameter Sweep
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
          Automatically search the hyperparameter space. The backend splits the dataset and trains separate SVMs for all combinations of $C$ and $\gamma$.
        </p>

        {error && (
          <div className="alert alert-danger" style={{ background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)', border: '1px solid rgba(239, 68, 68, 0.25)' }}>
            <AlertCircle size={18} />
            <div>{error}</div>
          </div>
        )}

        {/* Sweep Settings */}
        <div className="form-group">
          <label className="form-label">Sweep Kernel</label>
          <select 
            value={kernel} 
            onChange={(e) => setKernel(e.target.value)}
            disabled={loading}
            className="form-select"
          >
            <option value="rbf">RBF Kernel (Sweeps C & γ)</option>
            <option value="linear">Linear Kernel (Sweeps C only)</option>
          </select>
        </div>

        {/* Grid space descriptions */}
        <div style={{ fontSize: '0.8rem', background: 'rgba(0,0,0,0.15)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)', marginBottom: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>GRID SPACE COORDINATES:</div>
          <div>• C Values: <span style={{ fontWeight: 600, color: '#fff' }}>[{cValues.join(', ')}]</span></div>
          {kernel === 'rbf' && (
            <div>• Gamma (γ) Values: <span style={{ fontWeight: 600, color: '#fff' }}>[{gammaValues.join(', ')}]</span></div>
          )}
          <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginTop: '0.2rem' }}>
            Total separate models to train in pipeline: {kernel === 'rbf' ? cValues.length * gammaValues.length : cValues.length} runs.
          </div>
        </div>

        {samples.length < 5 ? (
          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px dashed var(--border-color)',
            borderRadius: '10px',
            padding: '1rem',
            textAlign: 'center',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)'
          }}>
            <AlertCircle size={20} style={{ margin: '0 auto 0.5rem', color: 'var(--warning)' }} />
            <p>Seeding is required before sweeping. Need at least 5 dataset samples.</p>
          </div>
        ) : (
          <button
            onClick={handleSweep}
            disabled={loading}
            className="btn btn-primary"
            style={{ width: '100%' }}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                <span>Executing Automated Sweep...</span>
              </>
            ) : (
              <>
                <Play size={18} fill="#fff" />
                <span>Run Grid Search Sweep</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Sweep Results Display */}
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BarChart3 size={20} color="var(--primary-glow)" />
            Hyperparameter Heatmap Results
          </h3>

          {!results && !loading && (
            <div style={{
              textAlign: 'center',
              padding: '4rem 1rem',
              color: 'var(--text-muted)',
              border: '1px dashed var(--border-color)',
              borderRadius: '12px',
              background: 'rgba(0,0,0,0.1)'
            }}>
              <BarChart3 size={40} style={{ margin: '0 auto 0.75rem', opacity: 0.4 }} />
              <p>Sweep Idle</p>
              <p style={{ fontSize: '0.75rem', marginTop: '0.2rem' }}>
                Run the grid search on the left to display parameter accuracy coordinates.
              </p>
            </div>
          )}

          {loading && (
            <div style={{
              textAlign: 'center',
              padding: '4rem 1rem',
              color: 'var(--text-muted)'
            }}>
              <span className="spinner spinner-lg" style={{ margin: '0 auto 1.5rem' }}></span>
              <p style={{ fontSize: '0.85rem' }}>Running validation runs on backend...</p>
            </div>
          )}

          {results && (
            <div style={{ animation: 'fadeIn 0.4s' }}>
              
              {/* Best configuration highlight */}
              {bestConfig && (
                <div style={{
                  background: 'rgba(16, 185, 129, 0.05)',
                  border: '1.5px solid var(--success)',
                  borderRadius: '10px',
                  padding: '1rem',
                  marginBottom: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem'
                }}>
                  <Award size={28} color="var(--success)" />
                  <div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 800, textTransform: 'uppercase', color: 'var(--success)' }}>
                      Optimal Parameter Configuration Found
                    </div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 700, marginTop: '0.1rem' }}>
                      C: {bestConfig.C} {kernel === 'rbf' ? `| γ: ${bestConfig.gamma}` : ''} ➔ Accuracy: {Math.round(bestConfig.accuracy * 100)}%
                    </div>
                  </div>
                </div>
              )}

              {/* Heatmap Grid */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: `80px repeat(${kernel === 'rbf' ? gammaValues.length : 1}, 1fr)`, gap: '0.5rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                  <div>C / γ</div>
                  {kernel === 'rbf' ? gammaValues.map(g => <div key={g}>γ={g}</div>) : <div>Linear</div>}
                </div>

                {cValues.map(c => (
                  <div key={c} style={{ display: 'grid', gridTemplateColumns: `80px repeat(${kernel === 'rbf' ? gammaValues.length : 1}, 1fr)`, gap: '0.5rem', alignItems: 'center' }}>
                    {/* C label */}
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textAlign: 'right', paddingRight: '0.5rem' }}>
                      C={c}
                    </div>

                    {/* Sweep blocks */}
                    {kernel === 'rbf' ? (
                      gammaValues.map(g => {
                        const r = results.find(x => x.C === c && x.gamma === g);
                        const acc = r ? r.accuracy : 0;
                        return (
                          <div
                            key={g}
                            style={{
                              background: getHeatmapColor(acc),
                              border: '1px solid var(--border-color)',
                              padding: '0.75rem 0',
                              borderRadius: '6px',
                              textAlign: 'center',
                              fontSize: '0.85rem',
                              fontWeight: 800,
                              color: acc > 0 ? '#fff' : 'var(--text-muted)'
                            }}
                            title={`C=${c}, gamma=${g}: Accuracy ${Math.round(acc * 100)}%`}
                          >
                            {r ? `${Math.round(acc * 100)}%` : '-'}
                          </div>
                        );
                      })
                    ) : (
                      (() => {
                        const r = results.find(x => x.C === c);
                        const acc = r ? r.accuracy : 0;
                        return (
                          <div
                            style={{
                              background: getHeatmapColor(acc),
                              border: '1px solid var(--border-color)',
                              padding: '0.75rem 0',
                              borderRadius: '6px',
                              textAlign: 'center',
                              fontSize: '0.85rem',
                              fontWeight: 800,
                              color: acc > 0 ? '#fff' : 'var(--text-muted)'
                            }}
                            title={`C=${c}: Accuracy ${Math.round(acc * 100)}%`}
                          >
                            {r ? `${Math.round(acc * 100)}%` : '-'}
                          </div>
                        );
                      })()
                    )}
                  </div>
                ))}
              </div>

            </div>
          )}
        </div>

        {results && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-color)', paddingTop: '0.75rem', marginTop: '1rem' }}>
            <CheckCircle size={14} />
            <span>Apply the optimal C and γ values on the Dashboard to maximize classification.</span>
          </div>
        )}
      </div>

    </div>
  );
}
