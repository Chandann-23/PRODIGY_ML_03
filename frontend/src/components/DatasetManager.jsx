import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Database, RefreshCw, Trash2, CheckCircle2, AlertCircle } from 'lucide-react';
import { extractFeatures, fileToImage, urlToImage } from '../utils/featureExtractor';

const DEMO_DOGS = [
  'https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=300&q=80',
  'https://images.unsplash.com/photo-1534361960057-19889db9621e?w=300&q=80',
  'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=300&q=80',
  'https://images.unsplash.com/photo-1517849845537-4d257902454a?w=300&q=80',
  'https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=300&q=80',
  'https://images.unsplash.com/photo-1537151608828-ea2b117b62e4?w=300&q=80',
  'https://images.unsplash.com/photo-1477884213984-b9710f22f493?w=300&q=80',
  'https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=300&q=80'
];

const DEMO_CATS = [
  'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300&q=80',
  'https://images.unsplash.com/photo-1519052537078-e6302a4968d4?w=300&q=80',
  'https://images.unsplash.com/photo-1495360010541-f48722b34f7d?w=300&q=80',
  'https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=300&q=80',
  'https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=300&q=80',
  'https://images.unsplash.com/photo-1533743983669-94fa5c4338ec?w=300&q=80',
  'https://images.unsplash.com/photo-1529778873920-4da4926a72c2?w=300&q=80',
  'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=300&q=80'
];

export default function DatasetManager({ samples, setSamples, onDatasetChange }) {
  const [label, setLabel] = useState(1); // 1 for Dog, -1 for Cat
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, text: '' });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  const fileInputRef = useRef(null);

  const handleReset = async () => {
    if (!window.confirm('Are you sure you want to clear the entire dataset?')) return;
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      await axios.delete('/api/samples');
      setSamples([]);
      onDatasetChange();
      setSuccess('Dataset cleared successfully.');
    } catch (err) {
      setError('Failed to clear dataset: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const seedDemoData = async () => {
    setLoading(true);
    setError('');
    setSuccess('');
    const total = DEMO_DOGS.length + DEMO_CATS.length;
    setProgress({ current: 0, total, text: 'Initializing neural network feature extractor...' });

    try {
      const seededSamples = [];

      // 1. Process dogs
      for (let i = 0; i < DEMO_DOGS.length; i++) {
        const url = DEMO_DOGS[i];
        setProgress({
          current: i + 1,
          total,
          text: `Processing dog image ${i + 1}/${DEMO_DOGS.length} (downloading & extracting 1024D features)...`
        });

        try {
          const img = await urlToImage(url);
          const features = await extractFeatures(img);
          seededSamples.push({
            imageUrl: url,
            label: 1,
            features,
            isDemo: true
          });
        } catch (imgErr) {
          console.warn(`Failed to seed dog image ${i + 1}`, imgErr);
        }
      }

      // 2. Process cats
      for (let i = 0; i < DEMO_CATS.length; i++) {
        const url = DEMO_CATS[i];
        setProgress({
          current: DEMO_DOGS.length + i + 1,
          total,
          text: `Processing cat image ${i + 1}/${DEMO_CATS.length} (downloading & extracting 1024D features)...`
        });

        try {
          const img = await urlToImage(url);
          const features = await extractFeatures(img);
          seededSamples.push({
            imageUrl: url,
            label: -1,
            features,
            isDemo: true
          });
        } catch (imgErr) {
          console.warn(`Failed to seed cat image ${i + 1}`, imgErr);
        }
      }

      if (seededSamples.length === 0) {
        throw new Error('All image feature extractions failed. Check CORS constraints.');
      }

      setProgress({ current: total, total, text: 'Uploading feature vectors to MongoDB Atlas...' });
      const res = await axios.post('/api/samples/bulk', { samples: seededSamples });
      
      setSuccess(`Seeded ${res.data.count} images successfully!`);
      onDatasetChange();
    } catch (err) {
      setError('Seeding failed: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
      setProgress({ current: 0, total: 0, text: '' });
    }
  };

  const handleFileUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setLoading(true);
    setError('');
    setSuccess('');
    const total = files.length;
    setProgress({ current: 0, total, text: 'Loading image processor...' });

    try {
      let uploadCount = 0;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setProgress({
          current: i + 1,
          total,
          text: `Processing: ${file.name} (Extracting TFJS features)...`
        });

        try {
          // Convert file to HTMLImageElement
          const img = await fileToImage(file);
          
          // Extract features via MobileNet
          const features = await extractFeatures(img);

          // Get image as base64 for preview storage
          const reader = new FileReader();
          const base64Data = await new Promise((resolve) => {
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
          });

          // Upload features and base64 preview
          await axios.post('/api/samples', {
            imageUrl: base64Data,
            label,
            features,
            isDemo: false
          });

          uploadCount++;
        } catch (fileErr) {
          console.error(`Error processing file ${file.name}:`, fileErr);
          setError(prev => `${prev}\nFailed to upload ${file.name}: ${fileErr.message}`);
        }
      }

      if (uploadCount > 0) {
        setSuccess(`Successfully uploaded ${uploadCount} images!`);
        onDatasetChange();
      }
    } catch (err) {
      setError('Upload failed: ' + err.message);
    } finally {
      setLoading(false);
      setProgress({ current: 0, total: 0, text: '' });
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const triggerFileSelect = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const dogCount = samples.filter(s => s.label === 1).length;
  const catCount = samples.filter(s => s.label === -1).length;

  return (
    <div className="dataset-grid">
      <div className="glass-card">
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Database size={20} color="var(--primary-glow)" />
          Dataset Management & Ingestion
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
          Ingest images into MongoDB. The browser extracts 1024-dimensional feature embeddings locally via MobileNet, ensuring high accuracy and zero backend workload.
        </p>

        {error && (
          <div className="alert alert-danger" style={{ whiteSpace: 'pre-line', background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)', border: '1px solid rgba(239, 68, 68, 0.25)' }}>
            <AlertCircle size={18} />
            <div>{error}</div>
          </div>
        )}

        {success && (
          <div className="alert" style={{ background: 'rgba(16, 185, 129, 0.1)', color: 'var(--success)', border: '1px solid rgba(16, 185, 129, 0.25)' }}>
            <CheckCircle2 size={18} />
            <div>{success}</div>
          </div>
        )}

        {/* Training Label Selector */}
        <div className="form-group">
          <label className="form-label">Upload Tag</label>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button
              onClick={() => setLabel(1)}
              className="btn"
              disabled={loading}
              style={{
                flex: 1,
                background: label === 1 ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
                borderColor: label === 1 ? 'var(--dog-color)' : 'var(--border-color)',
                color: label === 1 ? 'var(--dog-color)' : 'var(--text-secondary)',
                borderWidth: '1.5px'
              }}
            >
              Tag as Dog (+1)
            </button>
            <button
              onClick={() => setLabel(-1)}
              className="btn"
              disabled={loading}
              style={{
                flex: 1,
                background: label === -1 ? 'rgba(236, 72, 153, 0.15)' : 'transparent',
                borderColor: label === -1 ? 'var(--cat-color)' : 'var(--border-color)',
                color: label === -1 ? 'var(--cat-color)' : 'var(--text-secondary)',
                borderWidth: '1.5px'
              }}
            >
              Tag as Cat (-1)
            </button>
          </div>
        </div>

        {/* Dropzone */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          multiple
          accept="image/*"
          style={{ display: 'none' }}
        />
        
        <div 
          className="dropzone" 
          onClick={triggerFileSelect}
          style={{ pointerEvents: loading ? 'none' : 'auto', opacity: loading ? 0.6 : 1, marginBottom: '1.5rem' }}
        >
          <Upload size={36} color="var(--text-secondary)" />
          <p>
            Drag and drop images, or <span>browse files</span>
          </p>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Supports JPG, PNG (automatically downscaled to 224x224 and vectorized)
          </p>
        </div>

        {/* Progress Bar */}
        {loading && progress.total > 0 && (
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
              <span>{progress.text}</span>
              <span>{Math.round((progress.current / progress.total) * 100)}%</span>
            </div>
            <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
              <div 
                style={{ 
                  width: `${(progress.current / progress.total) * 100}%`, 
                  height: '100%', 
                  background: 'var(--primary-glow)', 
                  boxShadow: '0 0 8px var(--primary-glow)',
                  transition: 'width 0.2s ease-out' 
                }} 
              />
            </div>
          </div>
        )}

        {/* Actions panel */}
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          <button
            onClick={seedDemoData}
            disabled={loading}
            className="btn btn-primary"
            style={{ flex: 1, minWidth: '180px' }}
          >
            {loading && progress.total > 0 && progress.text.includes('Processing') ? (
              <span className="spinner"></span>
            ) : (
              <RefreshCw size={18} />
            )}
            Seed Demo Dataset (16 Images)
          </button>
          
          <button
            onClick={handleReset}
            disabled={loading || samples.length === 0}
            className="btn btn-danger"
            style={{ flex: 0.5, minWidth: '120px' }}
          >
            <Trash2 size={18} />
            Reset Data
          </button>
        </div>
      </div>

      {/* Dataset Grid Display */}
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>
            Active Dataset ({samples.length})
          </h3>
          <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.8rem' }}>
            <span style={{ color: 'var(--cat-color)', fontWeight: 600 }}>Cats: {catCount}</span>
            <span style={{ color: 'var(--text-muted)' }}>|</span>
            <span style={{ color: 'var(--dog-color)', fontWeight: 600 }}>Dogs: {dogCount}</span>
          </div>
        </div>

        {samples.length === 0 ? (
          <div style={{
            flexGrow: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-muted)',
            padding: '3rem 1rem',
            textAlign: 'center',
            border: '1px dashed var(--border-color)',
            borderRadius: '12px',
            background: 'rgba(0,0,0,0.1)'
          }}>
            <Database size={40} style={{ marginBottom: '1rem', opacity: 0.5 }} />
            <p style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>MongoDB Collection is Empty</p>
            <p style={{ fontSize: '0.8rem', marginTop: '0.2rem', maxWidth: '250px' }}>
              Click "Seed Demo Dataset" or drag custom files to ingest images and calculate features.
            </p>
          </div>
        ) : (
          <div className="thumbnail-grid" style={{ flexGrow: 1 }}>
            {samples.map((sample) => (
              <div 
                key={sample._id} 
                className={`thumbnail-wrapper ${sample.label === 1 ? 'dog' : 'cat'}`}
              >
                <img src={sample.imageUrl} alt={sample.label === 1 ? 'dog' : 'cat'} loading="lazy" />
                <div className={`thumbnail-label ${sample.label === 1 ? 'dog' : 'cat'}`}>
                  {sample.label === 1 ? 'Dog' : 'Cat'}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
