import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Eye, ShieldAlert, AlertTriangle, CheckCircle, HelpCircle } from 'lucide-react';
import { extractFeatures, fileToImage } from '../utils/featureExtractor';

export default function PredictorPanel({ activeModel, onPredictionResult, clearTestPoint }) {
  const [loading, setLoading] = useState(false);
  const [imgUrl, setImgUrl] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError('');
    setResult(null);
    clearTestPoint();

    try {
      // Create local object URL for preview
      const previewUrl = URL.createObjectURL(file);
      setImgUrl(previewUrl);

      // Extract image features locally in browser using MobileNet
      const img = await fileToImage(file);
      const features = await extractFeatures(img);

      // Submit feature vector to Express API for classification & outlier checks
      const res = await axios.post('/api/predict', { features });
      
      setResult(res.data);
      
      // Notify parent App.jsx so it can update the PCA plot with this test point
      onPredictionResult({
        x: res.data.projection.x,
        y: res.data.projection.y,
        label: res.data.label,
        imageUrl: previewUrl,
        isOutlier: res.data.isOutlier
      });

    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyze image: ' + err.message);
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const triggerFileSelect = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handleClear = () => {
    setResult(null);
    setImgUrl('');
    setError('');
    clearTestPoint();
  };

  if (!activeModel) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
        <ShieldAlert size={40} color="var(--warning)" style={{ marginBottom: '1rem', animation: 'pulse 2s infinite' }} />
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Inference Panel Locked</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.5rem', maxWidth: '400px', margin: '0.5rem auto 0' }}>
          No active model detected. You must seed the dataset and train/deploy an SVM model in the dashboard or tuning sweep before running predictions.
        </p>
      </div>
    );
  }

  const isDog = result?.label === 1;
  const confidencePercent = result ? Math.min(100, Math.round((Math.abs(result.score) / 1.5) * 100)) : 0; // rough confidence mapping relative to margin=1.0

  return (
    <div className="predictor-grid">
      
      {/* Upload & Preview Card */}
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Eye size={20} color="var(--primary-glow)" />
          Production Live Inference
        </h3>
        
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          style={{ display: 'none' }}
        />

        {!imgUrl ? (
          <div 
            className="dropzone" 
            onClick={triggerFileSelect}
            style={{ flexGrow: 1, display: 'flex', justifyContent: 'center' }}
          >
            <Upload size={36} color="var(--text-secondary)" />
            <p>
              Upload test image to <span>evaluate active SVM</span>
            </p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Supports any image (cats, dogs, or outliers to test data drift!)
            </p>
          </div>
        ) : (
          <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', position: 'relative' }}>
            <div style={{
              width: '100%',
              aspectRatio: '4/3',
              borderRadius: '12px',
              border: '1px solid var(--border-color)',
              overflow: 'hidden',
              background: '#020408',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <img 
                src={imgUrl} 
                alt="test-upload" 
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              />
            </div>
            
            <button 
              className="btn btn-secondary" 
              onClick={handleClear}
              disabled={loading}
              style={{ width: '100%' }}
            >
              Clear Image
            </button>
          </div>
        )}

        {loading && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.75rem',
            marginTop: '1.5rem',
            padding: '1rem',
            background: 'rgba(255,255,255,0.02)',
            borderRadius: '8px',
            border: '1px solid var(--border-color)'
          }}>
            <span className="spinner spinner-lg"></span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Extracting features (MobileNet) & classifying...
            </span>
          </div>
        )}

        {error && (
          <div className="alert alert-danger" style={{ marginTop: '1.5rem' }}>
            <AlertTriangle size={18} />
            <div>{error}</div>
          </div>
        )}
      </div>

      {/* Model Output Decision Card */}
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <div>
          <h3 style={{ fontSize: '1.10rem', fontWeight: 700, marginBottom: '1rem' }}>
            Model Inference Decision
          </h3>
          
          {!result && !loading && (
            <div style={{
              textAlign: 'center',
              padding: '3.5rem 1rem',
              color: 'var(--text-muted)',
              border: '1px dashed var(--border-color)',
              borderRadius: '12px',
              background: 'rgba(0,0,0,0.1)'
            }}>
              <HelpCircle size={40} style={{ margin: '0 auto 0.75rem', opacity: 0.4 }} />
              <p>Awaiting Test Upload</p>
              <p style={{ fontSize: '0.75rem', marginTop: '0.2rem' }}>
                Upload an image on the left. The model will calculate distance to decision hyperplane and evaluate classification score.
              </p>
            </div>
          )}

          {result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', animation: 'fadeIn 0.4s' }}>
              
              {/* Main Classification Result */}
              <div style={{
                background: isDog ? 'rgba(6, 182, 212, 0.08)' : 'rgba(236, 72, 153, 0.08)',
                border: `1.5px solid ${isDog ? 'var(--dog-color)' : 'var(--cat-color)'}`,
                borderRadius: '12px',
                padding: '1.25rem',
                textAlign: 'center',
                boxShadow: `0 0 15px ${isDog ? 'rgba(6, 182, 212, 0.1)' : 'rgba(236, 72, 153, 0.1)'}`
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '0.5rem' }}>
                  {isDog ? '🐶' : '🐱'}
                </div>
                <h4 style={{
                  fontSize: '1.5rem',
                  fontWeight: 800,
                  color: isDog ? 'var(--dog-color)' : 'var(--cat-color)',
                  textTransform: 'uppercase'
                }}>
                  {isDog ? 'Dog Detected' : 'Cat Detected'}
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  Classified using deployed active model: <strong>{activeModel.name}</strong>
                </p>
              </div>

              {/* Data Drift/Outlier Warning */}
              {result.isOutlier ? (
                <div className="alert alert-warning" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700 }}>
                    <AlertTriangle size={20} />
                    <span>⚠️ Data Drift / Outlier Alert</span>
                  </div>
                  <p style={{ fontSize: '0.75rem', lineHeight: 1.4 }}>
                    This image's feature vector is far from the training data centroids. Similarity is extremely low (Cats: {Math.round(result.similarities.cat * 100)}%, Dogs: {Math.round(result.similarities.dog * 100)}%). This mimics production data-drift alarms!
                  </p>
                </div>
              ) : (
                <div className="alert" style={{ background: 'rgba(16, 185, 129, 0.05)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
                  <CheckCircle size={16} />
                  <span>In-Distribution: Features match training data profiles.</span>
                </div>
              )}

              {/* SVM Plane Hyperplane margins info */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.8rem', background: 'rgba(0,0,0,0.15)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>SVM Decision Score ($f(x)$):</span>
                  <span style={{ fontWeight: 700, color: isDog ? 'var(--dog-color)' : 'var(--cat-color)' }}>
                    {result.score.toFixed(4)}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Margin Side:</span>
                  <span>{result.score >= 0 ? 'Dog (+1) Side' : 'Cat (-1) Side'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Distance to Margin boundary:</span>
                  <span style={{ fontWeight: 600 }}>
                    {Math.abs(result.score).toFixed(4)} units
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Cosine Similarity to Centroids:</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    🐱 {Math.round(result.similarities.cat * 100)}% | 🐶 {Math.round(result.similarities.dog * 100)}%
                  </span>
                </div>
              </div>

            </div>
          )}
        </div>

        {result && (
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.75rem' }}>
            The flashing indicator on the PCA canvas shows where this test point lies relative to the SVM decision hyperplane boundary.
          </div>
        )}
      </div>

    </div>
  );
}
