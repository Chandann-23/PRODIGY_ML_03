import express from 'express';
import { trainSVM, predictSVM, runPCA, projectTo2D } from '../utils/svmSolver.js';
import {
  findSamples,
  saveSample,
  insertManySamples,
  clearAllSamples,
  findActiveModel,
  findModels,
  saveModel,
  deployModel,
  deleteModel
} from '../utils/db.js';

const router = express.Router();

// Helper for Cosine Similarity
function cosineSimilarity(v1, v2) {
  if (!v1 || !v2 || v1.length === 0 || v2.length === 0) return 0;
  let dot = 0;
  let norm1 = 0;
  let norm2 = 0;
  const len = Math.min(v1.length, v2.length);
  for (let i = 0; i < len; i++) {
    dot += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  if (norm1 === 0 || norm2 === 0) return 0;
  return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

// 1. Get all samples (without feature vectors for listing)
router.get('/samples', async (req, res) => {
  try {
    const samples = await findSamples();
    // Exclude features to keep payloads small
    const formatted = samples.map(s => {
      const doc = s.toObject ? s.toObject() : s;
      const { features, ...rest } = doc;
      return rest;
    });
    res.json(formatted);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch samples', details: err.message });
  }
});

// 1b. Get all samples projected in 2D (using active model's PCA space)
router.get('/samples/projected', async (req, res) => {
  try {
    const model = await findActiveModel();
    if (!model) {
      return res.json([]);
    }

    const samples = await findSamples();
    
    // Setup PCA details
    const trainingMean = new Array(1024);
    for (let i = 0; i < 1024; i++) {
      trainingMean[i] = ((model.centroids?.cat?.[i] || 0) + (model.centroids?.dog?.[i] || 0)) / 2;
    }

    const pca = { projectionMatrix: model.pcaMatrix, mean: trainingMean };

    const projectedSamples = samples.map(s => {
      const doc = s.toObject ? s.toObject() : s;
      const [x, y] = projectTo2D(doc.features, pca);
      
      let isSupportVector = false;
      if (model.supportVectors && model.supportVectors.length > 0) {
        isSupportVector = model.supportVectors.some(sv => {
          return sv.features[0] === doc.features[0] && 
                 sv.features[10] === doc.features[10] && 
                 sv.features[100] === doc.features[100];
        });
      }

      return {
        _id: doc._id,
        imageUrl: doc.imageUrl,
        label: doc.label,
        isDemo: doc.isDemo,
        isSupportVector,
        x,
        y
      };
    });

    res.json(projectedSamples);
  } catch (err) {
    res.status(500).json({ error: 'Failed to project samples', details: err.message });
  }
});

// 2. Add single sample
router.post('/samples', async (req, res) => {
  const { imageUrl, label, features, isDemo } = req.body;
  try {
    const sample = await saveSample({ imageUrl, label, features, isDemo });
    res.status(201).json({ message: 'Sample saved successfully', id: sample._id });
  } catch (err) {
    res.status(400).json({ error: 'Failed to save sample', details: err.message });
  }
});

// 3. Bulk upload samples (for seeding the demo dataset)
router.post('/samples/bulk', async (req, res) => {
  const { samples } = req.body;
  try {
    if (!samples || !Array.isArray(samples) || samples.length === 0) {
      return res.status(400).json({ error: 'Samples must be a non-empty array' });
    }
    
    // Check if we already have demo data to avoid duplicates
    const count = await findSamples({ isDemo: true });
    if (count.length > 0) {
      return res.json({ message: 'Demo data already exists. Skipping seed.', count: count.length });
    }

    const inserted = await insertManySamples(samples);
    res.status(201).json({ message: `Successfully seeded ${inserted.length} samples`, count: inserted.length });
  } catch (err) {
    res.status(500).json({ error: 'Bulk seed failed', details: err.message });
  }
});

// 4. Delete all samples
router.delete('/samples', async (req, res) => {
  try {
    const result = await clearAllSamples();
    res.json({ message: 'All samples cleared', count: result.deletedCount });
  } catch (err) {
    res.status(500).json({ error: 'Failed to clear samples', details: err.message });
  }
});

// 5. Train SVM model (includes PCA & Split validation)
router.post('/train', async (req, res) => {
  const { kernel = 'linear', C = 1.0, gamma = 0.1, name } = req.body;
  const startTime = Date.now();

  try {
    const samples = await findSamples();
    if (samples.length < 4) {
      return res.status(400).json({ error: 'Need at least 4 samples to train a model!' });
    }

    // 1. Separate features and labels
    const X = samples.map(s => s.features);
    const y = samples.map(s => s.label);

    // 2. Perform PCA on the feature vectors to get projection matrix for 2D visualization
    const pca = runPCA(X, 2);

    // 3. Train-Test Split (80/20) if enough samples, otherwise use all for training
    const indices = Array.from({ length: X.length }, (_, i) => i);
    // Shuffle indices
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const splitIndex = Math.floor(X.length * 0.8);
    const useSplit = X.length >= 10;
    
    const trainIndices = useSplit ? indices.slice(0, splitIndex) : indices;
    const testIndices = useSplit ? indices.slice(splitIndex) : indices;

    const X_train = trainIndices.map(i => X[i]);
    const y_train = trainIndices.map(i => y[i]);
    const X_test = testIndices.map(i => X[i]);
    const y_test = testIndices.map(i => y[i]);

    // 4. Train the SVM model
    const trained = trainSVM(X_train, y_train, C, kernel, gamma, 2000);
    if (!trained) {
      return res.status(500).json({ error: 'SVM Solver failed to converge or return a model' });
    }

    // 5. Compute class centroids in 1024D for out-of-distribution drift checking
    const catVectors = X_train.filter((_, idx) => y_train[idx] === -1);
    const dogVectors = X_train.filter((_, idx) => y_train[idx] === 1);

    const computeCentroid = (vectors) => {
      if (vectors.length === 0) return new Array(1024).fill(0);
      const centroid = new Array(1024).fill(0);
      for (let i = 0; i < vectors.length; i++) {
        for (let j = 0; j < 1024; j++) {
          centroid[j] += vectors[i][j];
        }
      }
      for (let j = 0; j < 1024; j++) {
        centroid[j] /= vectors.length;
      }
      return centroid;
    };

    const catCentroid = computeCentroid(catVectors);
    const dogCentroid = computeCentroid(dogVectors);

    // 6. Evaluate model performance on validation set
    let tp = 0; // True Dog classified as Dog
    let tn = 0; // True Cat classified as Cat
    let fp = 0; // True Cat classified as Dog (false alarm)
    let fn = 0; // True Dog classified as Cat (miss)

    for (let i = 0; i < X_test.length; i++) {
      const pred = predictSVM(X_test[i], trained);
      const trueLabel = y_test[i];
      if (trueLabel === 1) {
        if (pred.label === 1) tp++;
        else fn++;
      } else {
        if (pred.label === -1) tn++;
        else fp++;
      }
    }

    const totalEval = X_test.length;
    const accuracy = totalEval > 0 ? (tp + tn) / totalEval : 0;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

    const trainingTimeMs = Date.now() - startTime;

    const modelName = name || `SVM ${kernel.toUpperCase()} (C=${C}${kernel === 'rbf' ? `, g=${gamma}` : ''})`;
    const versionNum = `v1.0.${Date.now().toString().slice(-4)}`;

    const newModel = await saveModel({
      name: modelName,
      version: versionNum,
      status: 'active',
      kernel,
      hyperparameters: { C, gamma },
      weights: trained.weights,
      bias: trained.bias,
      supportVectors: trained.supportVectors,
      metrics: {
        accuracy,
        precision,
        recall,
        f1,
        confusionMatrix: { tp, tn, fp, fn },
        trainingTimeMs
      },
      centroids: {
        cat: catCentroid,
        dog: dogCentroid
      },
      pcaMatrix: pca.projectionMatrix
    });

    // 8. Project all samples to 2D using our PCA components so the frontend can render immediately
    const projectedSamples = samples.map(s => {
      const doc = s.toObject ? s.toObject() : s;
      const [x, y] = projectTo2D(doc.features, pca);
      
      let isSupportVector = false;
      if (trained.supportVectors && trained.supportVectors.length > 0) {
        isSupportVector = trained.supportVectors.some(sv => {
          return sv.features[0] === doc.features[0] && 
                 sv.features[10] === doc.features[10] && 
                 sv.features[100] === doc.features[100];
        });
      }

      return {
        _id: doc._id,
        imageUrl: doc.imageUrl,
        label: doc.label,
        isDemo: doc.isDemo,
        isSupportVector,
        x,
        y
      };
    });

    res.status(201).json({
      message: 'Model trained and deployed successfully!',
      model: {
        _id: newModel._id,
        name: newModel.name,
        version: newModel.version,
        kernel: newModel.kernel,
        metrics: newModel.metrics,
        hyperparameters: newModel.hyperparameters,
        supportVectorCount: newModel.supportVectors.length,
        pcaMatrix: newModel.pcaMatrix,
        weights: newModel.weights,
        bias: newModel.bias,
        supportVectors: newModel.supportVectors,
        centroids: newModel.centroids
      },
      projectedSamples,
      pcaMatrix: pca.projectionMatrix
    });

  } catch (err) {
    res.status(500).json({ error: 'Training pipeline failed', details: err.message });
  }
});

// 6. Predict label for a feature vector (includes outlier detection)
router.post('/predict', async (req, res) => {
  const { features } = req.body;
  try {
    if (!features || !Array.isArray(features) || features.length !== 1024) {
      return res.status(400).json({ error: 'Must provide a 1024-dimensional feature vector' });
    }

    const model = await findActiveModel();
    if (!model) {
      return res.status(404).json({ error: 'No active SVM model found. Please train a model first!' });
    }

    // 1. Compute prediction score and label
    const pred = predictSVM(features, model);

    // 2. Perform Cosine Similarity outlier detection
    const catSim = cosineSimilarity(features, model.centroids.cat);
    const dogSim = cosineSimilarity(features, model.centroids.dog);
    const maxSim = Math.max(catSim, dogSim);

    // If similarity is below threshold, it's considered an out-of-distribution outlier (data drift)
    const DRIFT_THRESHOLD = 0.55; 
    const isOutlier = maxSim < DRIFT_THRESHOLD;

    // 3. Project test sample onto active model's 2D PCA space for visualization
    const trainingMean = new Array(1024);
    for (let i = 0; i < 1024; i++) {
      trainingMean[i] = ((model.centroids.cat[i] || 0) + (model.centroids.dog[i] || 0)) / 2;
    }
    const pca = { projectionMatrix: model.pcaMatrix, mean: trainingMean };

    const [x, y] = projectTo2D(features, pca);

    res.json({
      label: pred.label, // 1 for Dog, -1 for Cat
      score: pred.score, // Decision boundary distance
      isOutlier,
      similarities: {
        cat: catSim,
        dog: dogSim
      },
      projection: { x, y }
    });

  } catch (err) {
    res.status(500).json({ error: 'Prediction failed', details: err.message });
  }
});

// 7. Get the currently active model
router.get('/model/active', async (req, res) => {
  try {
    const model = await findActiveModel();
    if (!model) {
      return res.json(null);
    }
    
    // Format to match Mongoose select output
    const doc = model.toObject ? model.toObject() : model;
    const formatted = {
      _id: doc._id,
      name: doc.name,
      version: doc.version,
      kernel: doc.kernel,
      status: doc.status,
      hyperparameters: doc.hyperparameters,
      metrics: doc.metrics,
      createdAt: doc.createdAt,
      supportVectorCount: doc.supportVectors?.length || 0,
      centroids: doc.centroids,
      pcaMatrix: doc.pcaMatrix,
      weights: doc.weights,
      bias: doc.bias,
      supportVectors: doc.supportVectors
    };
    
    res.json(formatted);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch active model', details: err.message });
  }
});

// 8. Get all models (Registry)
router.get('/models', async (req, res) => {
  try {
    const models = await findModels();
    
    const formattedModels = models.map(m => {
      const doc = m.toObject ? m.toObject() : m;
      return {
        _id: doc._id,
        name: doc.name,
        version: doc.version,
        status: doc.status,
        kernel: doc.kernel,
        hyperparameters: doc.hyperparameters,
        metrics: doc.metrics,
        createdAt: doc.createdAt,
        supportVectorCount: doc.supportVectors?.length || 0
      };
    });
    
    res.json(formattedModels);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch model registry', details: err.message });
  }
});

// 9. Deploy a model from registry (sets to active)
router.post('/models/:id/deploy', async (req, res) => {
  try {
    const model = await deployModel(req.params.id);
    if (!model) {
      return res.status(404).json({ error: 'Model not found' });
    }
    
    res.json({ message: `Successfully deployed model ${model.version}`, model });
  } catch (err) {
    res.status(500).json({ error: 'Deployment failed', details: err.message });
  }
});

// 10. Delete a model
router.delete('/models/:id', async (req, res) => {
  try {
    const result = await deleteModel(req.params.id);
    if (result.deletedCount === 0) {
      return res.status(404).json({ error: 'Model not found' });
    }
    res.json({ message: `Deleted model successfully` });
  } catch (err) {
    res.status(500).json({ error: 'Deletion failed', details: err.message });
  }
});

// 11. Hyperparameter Sweep Grid Search
router.post('/train/sweep', async (req, res) => {
  const { kernel = 'linear', CValues = [0.1, 1.0, 10.0], gammaValues = [0.01, 0.1, 1.0] } = req.body;
  try {
    const samples = await findSamples();
    if (samples.length < 5) {
      return res.status(400).json({ error: 'Need at least 5 samples to run a grid search sweep!' });
    }

    const X = samples.map(s => s.features);
    const y = samples.map(s => s.label);

    // Shuffle and Split 80/20
    const indices = Array.from({ length: X.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const splitIndex = Math.floor(X.length * 0.8);
    const trainIndices = indices.slice(0, splitIndex);
    const testIndices = indices.slice(splitIndex);

    const X_train = trainIndices.map(i => X[i]);
    const y_train = trainIndices.map(i => y[i]);
    const X_test = testIndices.map(i => X[i]);
    const y_test = testIndices.map(i => y[i]);

    const results = [];

    // Loop over Hyperparameters
    for (const C of CValues) {
      if (kernel === 'linear') {
        const trained = trainSVM(X_train, y_train, C, 'linear', 0, 1000);
        if (trained) {
          let correct = 0;
          for (let i = 0; i < X_test.length; i++) {
            const pred = predictSVM(X_test[i], trained);
            if (pred.label === y_test[i]) correct++;
          }
          results.push({ C, gamma: 0, accuracy: correct / X_test.length });
        }
      } else {
        for (const gamma of gammaValues) {
          const trained = trainSVM(X_train, y_train, C, 'rbf', gamma, 1000);
          if (trained) {
            let correct = 0;
            for (let i = 0; i < X_test.length; i++) {
              const pred = predictSVM(X_test[i], trained);
              if (pred.label === y_test[i]) correct++;
            }
            results.push({ C, gamma, accuracy: correct / X_test.length });
          }
        }
      }
    }

    res.json({ kernel, results });

  } catch (err) {
    res.status(500).json({ error: 'Grid search sweep failed', details: err.message });
  }
});

export default router;
