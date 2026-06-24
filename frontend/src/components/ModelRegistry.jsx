import React, { useState } from 'react';
import axios from 'axios';
import { Eye, ShieldAlert, Award, Calendar, ToggleLeft, ToggleRight, Trash2, GitCompare, RefreshCcw } from 'lucide-react';

export default function ModelRegistry({ models, activeModel, onModelDeployed, onModelDeleted, refreshRegistry }) {
  const [comparingIds, setComparingIds] = useState([]);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleDeploy = async (id) => {
    try {
      await axios.post(`/api/models/${id}/deploy`);
      onModelDeployed();
    } catch (err) {
      alert('Failed to deploy model: ' + err.message);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this model run from the registry?')) return;
    try {
      await axios.delete(`/api/models/${id}`);
      onModelDeleted();
      // Remove from comparison if selected
      setComparingIds(prev => prev.filter(x => x !== id));
    } catch (err) {
      alert('Failed to delete model: ' + err.message);
    }
  };

  const handleCheckboxChange = (id) => {
    setComparingIds(prev => {
      if (prev.includes(id)) {
        return prev.filter(x => x !== id);
      } else {
        if (prev.length >= 2) {
          // Keep only the newest and the new selection (limit to 2)
          return [prev[1], id];
        }
        return [...prev, id];
      }
    });
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refreshRegistry();
    setIsRefreshing(false);
  };

  const formatDate = (dateStr) => {
    const d = new Date(dateStr);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const comparedModels = models.filter(m => comparingIds.includes(m._id));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Model Comparison Panel */}
      {comparedModels.length > 0 && (
        <div className="glass-card" style={{ border: '1.5px solid var(--primary-glow)', boxShadow: '0 0 20px rgba(99, 102, 241, 0.15)' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 800, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <GitCompare size={22} color="var(--primary-glow)" />
            Model Comparison Inspector ({comparedModels.length}/2)
          </h3>
          
          {comparedModels.length === 1 ? (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              Select one more model from the registry below to unlock side-by-side metric comparison.
            </p>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '2rem', overflowX: 'auto' }}>
              {comparedModels.map((m, idx) => (
                <div key={m._id} style={{ 
                  background: 'rgba(255,255,255,0.02)', 
                  border: '1px solid var(--border-color)', 
                  borderRadius: '12px',
                  padding: '1.25rem',
                  borderTop: `4px solid ${idx === 0 ? 'var(--dog-color)' : 'var(--cat-color)'}`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div>
                      <h4 style={{ fontWeight: 700, fontSize: '1rem' }}>{m.name}</h4>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Version: {m.version}</span>
                    </div>
                    <span className={`badge ${m.status === 'active' ? 'badge-active' : 'badge-archived'}`}>
                      {m.status}
                    </span>
                  </div>

                  {/* Core Metrics comparison grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', marginBottom: '1.5rem' }}>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.6rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>ACCURACY</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--success)' }}>
                        {Math.round(m.metrics.accuracy * 100)}%
                      </div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.6rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>F1-SCORE</div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--primary-glow)' }}>
                        {m.metrics.f1.toFixed(3)}
                      </div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.6rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>PRECISION</div>
                      <div style={{ fontSize: '1rem', fontWeight: 700 }}>
                        {m.metrics.precision.toFixed(3)}
                      </div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.6rem', borderRadius: '6px' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>RECALL</div>
                      <div style={{ fontSize: '1rem', fontWeight: 700 }}>
                        {m.metrics.recall.toFixed(3)}
                      </div>
                    </div>
                  </div>

                  {/* Latency and SV metrics */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem', marginBottom: '0.75rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Kernel Type:</span>
                      <span style={{ fontWeight: 600, textTransform: 'uppercase' }}>{m.kernel}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>C (Soft-margin):</span>
                      <span style={{ fontWeight: 600 }}>{m.hyperparameters.C}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Gamma:</span>
                      <span style={{ fontWeight: 600 }}>{m.kernel === 'rbf' ? m.hyperparameters.gamma : 'N/A'}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Support Vectors:</span>
                      <span style={{ fontWeight: 600, color: '#fbbf24' }}>{m.supportVectorCount} vectors</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Training Speed:</span>
                      <span style={{ fontWeight: 600 }}>{m.metrics.trainingTimeMs} ms</span>
                    </div>
                  </div>

                  {/* Confusion Matrix */}
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    <div style={{ textAlign: 'center', fontWeight: 700, marginBottom: '0.5rem' }}>Confusion Matrix</div>
                    <div className="matrix-container" style={{ margin: '0 auto', maxWidth: '280px' }}>
                      <div></div>
                      <div className="matrix-header-cell">Pred Cat</div>
                      <div className="matrix-header-cell">Pred Dog</div>
                      <div className="matrix-label-cell">True Cat</div>
                      <div className="matrix-cell tn">
                        <div className="val">{m.metrics.confusionMatrix.tn}</div>
                        <div style={{ fontSize: '0.55rem' }}>TN</div>
                      </div>
                      <div className="matrix-cell">
                        <div className="val">{m.metrics.confusionMatrix.fp}</div>
                        <div style={{ fontSize: '0.55rem' }}>FP</div>
                      </div>
                      <div className="matrix-label-cell">True Dog</div>
                      <div className="matrix-cell">
                        <div className="val">{m.metrics.confusionMatrix.fn}</div>
                        <div style={{ fontSize: '0.55rem' }}>FN</div>
                      </div>
                      <div className="matrix-cell tp">
                        <div className="val">{m.metrics.confusionMatrix.tp}</div>
                        <div style={{ fontSize: '0.55rem' }}>TP</div>
                      </div>
                    </div>
                  </div>

                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Models List */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Award size={22} color="var(--primary-glow)" />
              Model Warehouse & Deployment Registry
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.2rem' }}>
              Database history of trained SVM configurations. Check any two models to inspect performance side-by-side.
            </p>
          </div>
          
          <button 
            className="btn btn-secondary" 
            onClick={handleRefresh} 
            disabled={isRefreshing}
            style={{ padding: '0.5rem 1rem' }}
          >
            <RefreshCcw size={16} className={isRefreshing ? 'spinner' : ''} />
            Refresh
          </button>
        </div>

        {models.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '3rem 1rem',
            border: '1px dashed var(--border-color)',
            borderRadius: '12px',
            color: 'var(--text-muted)'
          }}>
            <ShieldAlert size={36} style={{ marginBottom: '0.75rem', opacity: 0.5 }} />
            <p style={{ fontWeight: 600 }}>No Models Registered</p>
            <p style={{ fontSize: '0.8rem', marginTop: '0.2rem' }}>
              Train your first SVM model in the Dashboard or Tuning sweep view!
            </p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '0.75rem 1rem', width: '40px' }}>Comp</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Model Details</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Hyperparams</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Support Vectors</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Validation Split Accuracy</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Status</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => {
                  const isActive = m.status === 'active';
                  const isChecked = comparingIds.includes(m._id);
                  return (
                    <tr 
                      key={m._id} 
                      style={{ 
                        borderBottom: '1px solid var(--border-color)',
                        background: isActive ? 'rgba(99, 102, 241, 0.02)' : 'transparent',
                        transition: 'background 0.2s'
                      }}
                      className="table-row-hover"
                    >
                      <td style={{ padding: '1rem' }}>
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => handleCheckboxChange(m._id)}
                          style={{ cursor: 'pointer', width: '16px', height: '16px' }}
                        />
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <div style={{ fontWeight: 700, color: isActive ? '#fff' : 'var(--text-primary)' }}>{m.name}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem', marginTop: '0.2rem' }}>
                          <Calendar size={12} />
                          {formatDate(m.createdAt)} ({m.version})
                        </div>
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <div style={{ textTransform: 'uppercase', fontWeight: 600, fontSize: '0.75rem' }}>Kernel: {m.kernel}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                          C={m.hyperparameters.C}{m.kernel === 'rbf' ? `, g=${m.hyperparameters.gamma}` : ''}
                        </div>
                      </td>
                      <td style={{ padding: '1rem', fontWeight: 600, color: '#fbbf24' }}>
                        {m.supportVectorCount} vectors
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <div style={{ fontWeight: 800, fontSize: '1rem', color: 'var(--success)' }}>
                            {Math.round(m.metrics.accuracy * 100)}%
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                            (F1: {m.metrics.f1.toFixed(2)})
                          </div>
                        </div>
                      </td>
                      <td style={{ padding: '1rem' }}>
                        <span className={`badge ${isActive ? 'badge-active' : 'badge-archived'}`}>
                          {m.status}
                        </span>
                      </td>
                      <td style={{ padding: '1rem', textAlign: 'right' }}>
                        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                          <button
                            onClick={() => handleDeploy(m._id)}
                            disabled={isActive}
                            className={`btn ${isActive ? 'btn-secondary' : 'btn-primary'}`}
                            style={{ 
                              padding: '0.4rem 0.75rem', 
                              fontSize: '0.8rem', 
                              opacity: isActive ? 0.4 : 1,
                              cursor: isActive ? 'default' : 'pointer'
                            }}
                            title={isActive ? 'Currently deployed' : 'Set active model'}
                          >
                            {isActive ? <ToggleRight size={16} /> : <ToggleLeft size={16} />}
                            {isActive ? 'Active' : 'Deploy'}
                          </button>
                          
                          <button
                            onClick={() => handleDelete(m._id)}
                            className="btn btn-danger"
                            style={{ padding: '0.4rem', borderRadius: '8px' }}
                            title="Delete run"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}
