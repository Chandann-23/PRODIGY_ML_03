/**
 * Pure JavaScript implementation of Support Vector Machine (SVM) training via SMO
 * and Principal Component Analysis (PCA) for dimensionality reduction.
 */

/**
 * Trains a support vector machine using Sequential Minimal Optimization (SMO)
 * 
 * @param {Array<Array<Number>>} X - Feature vectors (N x D)
 * @param {Array<Number>} y - Labels (N), containing values of -1 or 1
 * @param {Number} C - Regularization parameter
 * @param {String} kernelType - 'linear' or 'rbf'
 * @param {Number} gamma - RBF kernel parameter
 * @param {Number} maxIter - Maximum outer iterations
 * @returns {Object} Trained model parameters
 */
export function trainSVM(X, y, C, kernelType, gamma, maxIter = 1000) {
  const N = X.length;
  if (N === 0) return null;
  const D = X[0].length;

  const alphas = new Array(N).fill(0);
  let b = 0;
  const tol = 1e-4;

  const kernel = (x1, x2) => {
    if (kernelType === 'linear') {
      let sum = 0;
      for (let i = 0; i < D; i++) sum += x1[i] * x2[i];
      return sum;
    } else {
      let sum = 0;
      for (let i = 0; i < D; i++) {
        const diff = x1[i] - x2[i];
        sum += diff * diff;
      }
      return Math.exp(-gamma * sum);
    }
  };

  // Precompute kernel matrix for speed
  const K = [];
  for (let i = 0; i < N; i++) {
    K.push(new Array(N));
  }
  for (let i = 0; i < N; i++) {
    for (let j = i; j < N; j++) {
      const val = kernel(X[i], X[j]);
      K[i][j] = val;
      K[j][i] = val;
    }
  }

  // SMO solver main loop
  let iter = 0;
  let passes = 0;
  const maxPasses = 10; // Number of passes with no changes before terminating

  while (passes < maxPasses && iter < maxIter) {
    let numChangedAlphas = 0;
    for (let i = 0; i < N; i++) {
      // Calculate f(x_i)
      let f_xi = b;
      for (let k = 0; k < N; k++) {
        f_xi += alphas[k] * y[k] * K[k][i];
      }
      const E_i = f_xi - y[i];

      // Check if alpha_i violates KKT conditions
      if ((y[i] * E_i < -tol && alphas[i] < C) || (y[i] * E_i > tol && alphas[i] > 0)) {
        // Select random j != i
        let j = Math.floor(Math.random() * (N - 1));
        if (j >= i) j++;

        // Calculate f(x_j)
        let f_xj = b;
        for (let k = 0; k < N; k++) {
          f_xj += alphas[k] * y[k] * K[k][j];
        }
        const E_j = f_xj - y[j];

        const alpha_i_old = alphas[i];
        const alpha_j_old = alphas[j];

        // Compute L and H (bounds for alpha_j)
        let L = 0;
        let H = C;
        if (y[i] !== y[j]) {
          L = Math.max(0, alphas[j] - alphas[i]);
          H = Math.min(C, C + alphas[j] - alphas[i]);
        } else {
          L = Math.max(0, alphas[i] + alphas[j] - C);
          H = Math.min(C, alphas[i] + alphas[j]);
        }

        if (L === H) continue;

        // Compute eta
        const eta = 2 * K[i][j] - K[i][i] - K[j][j];
        if (eta >= 0) continue;

        // Update alpha_j
        let alpha_j_new = alpha_j_old - (y[j] * (E_i - E_j)) / eta;
        
        // Clip alpha_j
        if (alpha_j_new > H) alpha_j_new = H;
        else if (alpha_j_new < L) alpha_j_new = L;

        if (Math.abs(alpha_j_new - alpha_j_old) < 1e-5) continue;

        // Update alpha_i
        const alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new);
        
        alphas[i] = alpha_i_new;
        alphas[j] = alpha_j_new;

        // Update bias b
        const b1 = b - E_i - y[i] * (alpha_i_new - alpha_i_old) * K[i][i] - y[j] * (alpha_j_new - alpha_j_old) * K[i][j];
        const b2 = b - E_j - y[i] * (alpha_i_new - alpha_i_old) * K[i][j] - y[j] * (alpha_j_new - alpha_j_old) * K[j][j];

        if (0 < alpha_i_new && alpha_i_new < C) {
          b = b1;
        } else if (0 < alpha_j_new && alpha_j_new < C) {
          b = b2;
        } else {
          b = (b1 + b2) / 2;
        }

        numChangedAlphas++;
      }
    }

    iter++;
    if (numChangedAlphas === 0) {
      passes++;
    } else {
      passes = 0;
    }
  }

  // Calculate weights for linear kernel
  let weights = [];
  if (kernelType === 'linear') {
    weights = new Array(D).fill(0);
    for (let j = 0; j < D; j++) {
      for (let i = 0; i < N; i++) {
        weights[j] += alphas[i] * y[i] * X[i][j];
      }
    }
  }

  // Extract support vectors
  const supportVectors = [];
  for (let i = 0; i < N; i++) {
    if (alphas[i] > 1e-5) {
      supportVectors.push({
        features: X[i],
        label: y[i],
        alpha: alphas[i]
      });
    }
  }

  return {
    kernel: kernelType,
    hyperparameters: { C, gamma },
    weights,
    bias: b,
    supportVectors,
    iterations: iter
  };
}

/**
 * Predicts the score/class of a feature vector given a trained model
 */
export function predictSVM(features, model) {
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

  // For RBF or general support vector prediction
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

/**
 * Computes Principal Component Analysis (PCA) to find projection directions
 * that map D-dimensional features to 2D.
 * Uses a matrix-free Power Iteration method with deflation.
 * 
 * @param {Array<Array<Number>>} X - Centered/Raw data matrix (N x D)
 * @param {Number} numComponents - Number of principal components (default 2)
 * @returns {Object} PCA mean vector and 2 x D projection matrix
 */
export function runPCA(X, numComponents = 2) {
  const N = X.length;
  if (N === 0) return { projectionMatrix: [], mean: [] };
  const D = X[0].length;

  // 1. Compute Mean
  const mean = new Array(D).fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < D; j++) {
      mean[j] += X[i][j];
    }
  }
  for (let j = 0; j < D; j++) {
    mean[j] /= N;
  }

  // 2. Center the data
  const X_centered = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(D);
    for (let j = 0; j < D; j++) {
      row[j] = X[i][j] - mean[j];
    }
    X_centered.push(row);
  }

  // 3. Matrix-free Power Iteration to find top eigenvectors
  // C = (1/N) * X_centered^T * X_centered
  // C * v = (1/N) * X_centered^T * (X_centered * v)
  const eigenvectors = [];
  const eigenvalues = [];

  for (let k = 0; k < numComponents; k++) {
    let v = new Array(D);
    let norm = 0;
    
    // Seed with deterministic starting values to prevent minor chart flipping
    for (let j = 0; j < D; j++) {
      v[j] = Math.sin(j + k + 1); 
      norm += v[j] * v[j];
    }
    norm = Math.sqrt(norm);
    for (let j = 0; j < D; j++) {
      v[j] /= norm;
    }

    let lambda = 0;
    const maxIterations = 80;
    const tol = 1e-5;

    for (let iter = 0; iter < maxIterations; iter++) {
      // u = X_centered * v (size N)
      const u = new Array(N).fill(0);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < D; j++) {
          u[i] += X_centered[i][j] * v[j];
        }
      }

      // w = (1/N) * X_centered^T * u (size D)
      const w = new Array(D).fill(0);
      for (let j = 0; j < D; j++) {
        for (let i = 0; i < N; i++) {
          w[j] += X_centered[i][j] * u[i];
        }
        w[j] /= N;
      }

      // Deflation: w = A_k * v = C * v - sum_{i=0}^{k-1} lambda_i * e_i * (e_i^T * v)
      for (let i = 0; i < k; i++) {
        const e_i = eigenvectors[i];
        const lambda_i = eigenvalues[i];
        let dot = 0;
        for (let j = 0; j < D; j++) {
          dot += e_i[j] * v[j];
        }
        for (let j = 0; j < D; j++) {
          w[j] -= lambda_i * e_i[j] * dot;
        }
      }

      // Compute new norm
      let nextNorm = 0;
      for (let j = 0; j < D; j++) {
        nextNorm += w[j] * w[j];
      }
      nextNorm = Math.sqrt(nextNorm);

      if (nextNorm < 1e-9) break;

      // Estimate eigenvalue (Rayleigh quotient)
      let dot_v_w = 0;
      for (let j = 0; j < D; j++) {
        dot_v_w += v[j] * w[j];
      }
      const newLambda = dot_v_w;

      // Check convergence
      let diff = 0;
      for (let j = 0; j < D; j++) {
        const d = (w[j] / nextNorm) - v[j];
        diff += d * d;
      }
      diff = Math.sqrt(diff);

      // Update v
      for (let j = 0; j < D; j++) {
        v[j] = w[j] / nextNorm;
      }
      lambda = newLambda;

      if (diff < tol) break;
    }

    eigenvectors.push(v);
    eigenvalues.push(lambda);
  }

  // projectionMatrix is a 2 x D matrix
  return {
    projectionMatrix: eigenvectors,
    mean: mean
  };
}

/**
 * Projects a high dimensional vector to 2D using a projection matrix and mean vector
 */
export function projectTo2D(features, pca) {
  const { projectionMatrix, mean } = pca;
  if (!projectionMatrix || projectionMatrix.length < 2) return [0, 0];
  
  const D = features.length;
  const centered = new Array(D);
  for (let i = 0; i < D; i++) {
    centered[i] = features[i] - (mean[i] || 0);
  }

  const x = projectionMatrix[0].reduce((acc, val, idx) => acc + val * centered[idx], 0);
  const y = projectionMatrix[1].reduce((acc, val, idx) => acc + val * centered[idx], 0);

  return [x, y];
}
