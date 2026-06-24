import React, { useRef, useEffect, useState } from 'react';
import { predictSVM } from '../../../backend/utils/svmSolver.js'; // We can replicate the simple predictSVM function here to keep frontend standalone

// Simple replica of prediction logic for browser-side rendering to avoid API overhead
function localPredictSVM(features, model) {
  const { kernel, hyperparameters, bias, supportVectors, weights } = model;
  
  if (kernel === 'linear' && weights && weights.length > 0) {
    let score = bias;
    for (let i = 0; i < features.length; i++) {
      score += weights[i] * features[i];
    }
    return {
      score,
      label: score >= 0 ? 1 : -1
    };
  }

  let score = bias;
  const D = features.length;
  for (let i = 0; i < supportVectors.length; i++) {
    const sv = supportVectors[i];
    
    let kernelVal = 0;
    if (kernel === 'linear') {
      let sum = 0;
      for (let d = 0; d < D; d++) sum += sv.features[d] * features[d];
      kernelVal = sum;
    } else {
      let sum = 0;
      for (let d = 0; d < D; d++) {
        const diff = sv.features[d] - features[d];
        sum += diff * diff;
      }
      kernelVal = Math.exp(-hyperparameters.gamma * sum);
    }
    
    score += sv.alpha * sv.label * kernelVal;
  }

  return {
    score,
    label: score >= 0 ? 1 : -1
  };
}

export default function PCABoundaryCanvas({ model, samples, testPoint }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !model || !samples || samples.length === 0) return;

    const ctx = canvas.getContext('2d');
    
    // Set size based on container width
    const rect = containerRef.current.getBoundingClientRect();
    const size = Math.min(rect.width, 550);
    canvas.width = size;
    canvas.height = size;

    const W = canvas.width;
    const H = canvas.height;
    const padding = 50;

    // 1. Find bounding box of 2D data points (x, y)
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    samples.forEach(s => {
      if (s.x < minX) minX = s.x;
      if (s.x > maxX) maxX = s.x;
      if (s.y < minY) minY = s.y;
      if (s.y > maxY) maxY = s.y;
    });

    if (testPoint) {
      if (testPoint.x < minX) minX = testPoint.x;
      if (testPoint.x > maxX) maxX = testPoint.x;
      if (testPoint.y < minY) minY = testPoint.y;
      if (testPoint.y > maxY) maxY = testPoint.y;
    }

    // Add margin to bounding box
    const dx = maxX - minX || 2;
    const dy = maxY - minY || 2;
    minX -= dx * 0.15;
    maxX += dx * 0.15;
    minY -= dy * 0.15;
    maxY += dy * 0.15;

    // Coordinate conversion helpers
    const toCanvasX = (rx) => padding + ((rx - minX) / (maxX - minX)) * (W - 2 * padding);
    const toCanvasY = (ry) => H - padding - ((ry - minY) / (maxY - minY)) * (H - 2 * padding);
    const toRealX = (cx) => minX + ((cx - padding) / (W - 2 * padding)) * (maxX - minX);
    const toRealY = (cy) => minY + ((H - padding - cy) / (H - 2 * padding)) * (maxY - minY);

    // 2. Draw Decision Boundary Heatmap (evaluate a grid of 2D points)
    // We reconstruct the 1024D representation of each 2D point and run localPredictSVM
    const pcaMatrix = model.pcaMatrix;
    if (!pcaMatrix || !pcaMatrix[0] || !pcaMatrix[1]) {
      console.warn('PCABoundaryCanvas: pcaMatrix is missing or incomplete in model data.', model);
      return;
    }
    // Calculate original feature mean from saved centroids
    const mean = new Array(1024);
    for (let i = 0; i < 1024; i++) {
      mean[i] = ((model.centroids?.cat?.[i] || 0) + (model.centroids?.dog?.[i] || 0)) / 2;
    }

    const gridResolution = 80; // Grid resolution (80x80 pixels)
    const cellW = W / gridResolution;
    const cellH = H / gridResolution;

    for (let gx = 0; gx < gridResolution; gx++) {
      for (let gy = 0; gy < gridResolution; gy++) {
        const cx = gx * cellW + cellW / 2;
        const cy = gy * cellH + cellH / 2;

        const rx = toRealX(cx);
        const ry = toRealY(cy);

        // Reconstruct 1024D vector: x_1024 = mean + z1 * p1 + z2 * p2
        const reconstructed = new Array(1024);
        for (let d = 0; d < 1024; d++) {
          reconstructed[d] = mean[d] + rx * (pcaMatrix[0]?.[d] || 0) + ry * (pcaMatrix[1]?.[d] || 0);
        }

        const pred = localPredictSVM(reconstructed, model);
        
        // Coloring logic based on SVM score
        // Score > 0 is Dog territory, Score < 0 is Cat territory
        // Draw margin zone where -1 <= score <= 1
        let baseColor = [3, 7, 15]; // bg-color
        
        if (pred.score >= 0) {
          // Dog territory (Cyan tint)
          const alpha = Math.min(0.2, pred.score * 0.08);
          ctx.fillStyle = `rgba(6, 182, 212, ${alpha})`;
        } else {
          // Cat territory (Magenta tint)
          const alpha = Math.min(0.2, Math.abs(pred.score) * 0.08);
          ctx.fillStyle = `rgba(236, 72, 153, ${alpha})`;
        }

        ctx.fillRect(gx * cellW, gy * cellH, cellW + 0.5, cellH + 0.5);

        // Draw decision boundary contour line (where score is close to 0)
        // If absolute score is very close to 0, shade it white/violet
        if (Math.abs(pred.score) < 0.08) {
          ctx.fillStyle = 'rgba(99, 102, 241, 0.4)';
          ctx.fillRect(gx * cellW, gy * cellH, cellW + 0.5, cellH + 0.5);
        }
        
        // Draw margin lines (score is close to 1 or -1)
        if (Math.abs(Math.abs(pred.score) - 1.0) < 0.05) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
          ctx.fillRect(gx * cellW, gy * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    }

    // 3. Draw Axis
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, H - padding);
    ctx.lineTo(W - padding, H - padding);
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, H - padding);
    ctx.stroke();

    // Axis titles
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '600 10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText('First Principal Component (PC1)', W - padding, H - padding + 15);
    ctx.save();
    ctx.translate(padding - 15, padding);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Second Principal Component (PC2)', 0, 0);
    ctx.restore();

    // 4. Draw training samples
    samples.forEach(s => {
      const cx = toCanvasX(s.x);
      const cy = toCanvasY(s.y);
      const isDog = s.label === 1;

      // Glow effect if Support Vector
      if (s.isSupportVector) {
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#fbbf24'; // Gold glow for support vectors
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.shadowBlur = 0; // reset glow
      }

      // Draw point core
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fillStyle = isDog ? '#06b6d4' : '#ec4899';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });

    // 5. Draw test point if exists
    if (testPoint) {
      const cx = toCanvasX(testPoint.x);
      const cy = toCanvasY(testPoint.y);
      const isDog = testPoint.label === 1;

      // Draw large flashing outer ring
      ctx.shadowBlur = 15;
      ctx.shadowColor = testPoint.isOutlier ? '#f59e0b' : (isDog ? '#06b6d4' : '#ec4899');
      ctx.strokeStyle = testPoint.isOutlier ? '#f59e0b' : (isDog ? '#06b6d4' : '#ec4899');
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, 12, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw core
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
      ctx.fill();
    }

  }, [model, samples, testPoint]);

  // Handle Mouse Over to show Tooltip
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || !samples || samples.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const W = canvas.width;
    const H = canvas.height;
    const padding = 50;

    // Get bounding box again to calculate inverse coordinates
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    samples.forEach(s => {
      if (s.x < minX) minX = s.x;
      if (s.x > maxX) maxX = s.x;
      if (s.y < minY) minY = s.y;
      if (s.y > maxY) maxY = s.y;
    });
    if (testPoint) {
      if (testPoint.x < minX) minX = testPoint.x;
      if (testPoint.x > maxX) maxX = testPoint.x;
      if (testPoint.y < minY) minY = testPoint.y;
      if (testPoint.y > maxY) maxY = testPoint.y;
    }
    const dx = maxX - minX || 2;
    const dy = maxY - minY || 2;
    minX -= dx * 0.15;
    maxX += dx * 0.15;
    minY -= dy * 0.15;
    maxY += dy * 0.15;

    const toCanvasX = (rx) => padding + ((rx - minX) / (maxX - minX)) * (W - 2 * padding);
    const toCanvasY = (ry) => H - padding - ((ry - minY) / (maxY - minY)) * (H - 2 * padding);

    // Find nearest point within 10 pixels radius
    let nearest = null;
    let minDist = 12;

    samples.forEach(s => {
      const scx = toCanvasX(s.x);
      const scy = toCanvasY(s.y);
      const dist = Math.sqrt((cx - scx) ** 2 + (cy - scy) ** 2);
      if (dist < minDist) {
        minDist = dist;
        nearest = s;
      }
    });

    if (testPoint) {
      const scx = toCanvasX(testPoint.x);
      const scy = toCanvasY(testPoint.y);
      const dist = Math.sqrt((cx - scx) ** 2 + (cy - scy) ** 2);
      if (dist < minDist) {
        minDist = dist;
        nearest = { ...testPoint, _id: 'test', isTestPoint: true };
      }
    }

    if (nearest) {
      setHoveredPoint(nearest);
      setTooltipPos({ x: e.clientX - rect.left + 15, y: e.clientY - rect.top - 70 });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative' }}>
      <div style={{ alignSelf: 'flex-start', marginBottom: '1rem', width: '100%' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--primary-glow)', boxShadow: '0 0 8px var(--primary-glow)' }}></span>
          2D PCA Feature Boundary Projection
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
          Visualizing high-dimensional embeddings. Dots represent images. Shaded fields represent classification boundaries.
        </p>
      </div>

      <div ref={containerRef} style={{ width: '100%', position: 'relative', background: '#03050a', borderRadius: '12px', border: '1px solid var(--border-color)', overflow: 'hidden' }}>
        <canvas
          ref={canvasRef}
          style={{ display: 'block', cursor: hoveredPoint ? 'pointer' : 'default' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />

        {/* Hover Tooltip */}
        {hoveredPoint && (
          <div style={{
            position: 'absolute',
            left: `${tooltipPos.x}px`,
            top: `${tooltipPos.y}px`,
            background: 'rgba(5, 8, 20, 0.95)',
            border: `1px solid ${hoveredPoint.isSupportVector ? '#fbbf24' : 'var(--border-color)'}`,
            borderRadius: '10px',
            padding: '0.5rem',
            pointerEvents: 'none',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
            zIndex: 10,
            animation: 'fadeIn 0.15s ease-out'
          }}>
            {hoveredPoint.imageUrl && (
              <img 
                src={hoveredPoint.imageUrl} 
                alt="thumbnail" 
                style={{ width: '48px', height: '48px', objectFit: 'cover', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)' }}
              />
            )}
            <div>
              <div style={{ fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', color: hoveredPoint.label === 1 ? 'var(--dog-color)' : 'var(--cat-color)' }}>
                {hoveredPoint.isTestPoint ? 'Test Image' : (hoveredPoint.label === 1 ? 'Dog' : 'Cat')}
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>
                {hoveredPoint.isSupportVector ? '🌟 Support Vector' : 'Dataset Sample'}
              </div>
              {hoveredPoint.isTestPoint && hoveredPoint.isOutlier && (
                <div style={{ fontSize: '0.65rem', color: 'var(--warning)', fontWeight: 700 }}>
                  ⚠️ Data Drift Outlier
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', marginTop: '1.25rem', fontSize: '0.8rem', justifyContent: 'center' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'var(--cat-color)', border: '1px solid #fff' }}></span>
          Cats
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'var(--dog-color)', border: '1px solid #fff' }}></span>
          Dogs
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ width: '12px', height: '12px', borderRadius: '50%', border: '2px solid #fbbf24', background: 'transparent' }}></span>
          Support Vectors
        </span>
        {testPoint && (
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#fff', boxShadow: '0 0 8px #fff' }}></span>
            Active Test Point
          </span>
        )}
      </div>
    </div>
  );
}
