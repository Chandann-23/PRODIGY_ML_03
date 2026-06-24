import fs from 'fs';
import path from 'path';
import Sample from '../models/Sample.js';
import SVMModel from '../models/SVMModel.js';

let isMongoConnected = false;
const fallbackFile = path.resolve('database_fallback.json');

// Local in-memory store in case of fallback
let localData = {
  samples: [],
  models: []
};

// Load local fallback data from file if it exists
try {
  if (fs.existsSync(fallbackFile)) {
    const raw = fs.readFileSync(fallbackFile, 'utf8');
    if (raw.trim()) {
      localData = JSON.parse(raw);
      console.log(`[Database Fallback] Loaded ${localData.samples.length} samples and ${localData.models.length} models from local JSON.`);
    }
  }
} catch (e) {
  console.warn('[Database Fallback] Failed to load local database fallback file:', e.message);
}

function saveFallback() {
  try {
    fs.writeFileSync(fallbackFile, JSON.stringify(localData, null, 2), 'utf8');
  } catch (e) {
    console.error('[Database Fallback] Failed to save database to JSON file:', e.message);
  }
}

export function setMongoConnected(status) {
  isMongoConnected = status;
  console.log(`[Database Mode] Switched active database engine to: ${status ? 'MongoDB Atlas / Local Server' : 'Local JSON File File-DB'}`);
}

export function getMongoConnected() {
  return isMongoConnected;
}

export async function findSamples(filter = {}) {
  if (isMongoConnected) {
    return await Sample.find(filter);
  }
  
  // Apply simple filter support for local array
  if (filter.isDemo !== undefined) {
    return localData.samples.filter(s => s.isDemo === filter.isDemo);
  }
  return localData.samples;
}

export async function countSamples(filter = {}) {
  if (isMongoConnected) {
    return await Sample.countDocuments(filter);
  }
  return (await findSamples(filter)).length;
}

export async function saveSample(sampleData) {
  if (isMongoConnected) {
    const sample = new Sample(sampleData);
    return await sample.save();
  }
  const newSample = {
    _id: 'sample_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
    ...sampleData,
    createdAt: new Date().toISOString()
  };
  localData.samples.push(newSample);
  saveFallback();
  return newSample;
}

export async function insertManySamples(samplesArray) {
  if (isMongoConnected) {
    return await Sample.insertMany(samplesArray);
  }
  const inserted = samplesArray.map(s => ({
    _id: 'sample_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
    ...s,
    createdAt: new Date().toISOString()
  }));
  localData.samples = [...localData.samples, ...inserted];
  saveFallback();
  return inserted;
}

export async function clearAllSamples() {
  if (isMongoConnected) {
    return await Sample.deleteMany({});
  }
  const count = localData.samples.length;
  localData.samples = [];
  saveFallback();
  return { deletedCount: count };
}

export async function findActiveModel() {
  if (isMongoConnected) {
    return await SVMModel.findOne({ status: 'active' });
  }
  return localData.models.find(m => m.status === 'active') || null;
}

export async function findModels() {
  if (isMongoConnected) {
    return await SVMModel.find().sort({ createdAt: -1 });
  }
  return [...localData.models].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
}

export async function saveModel(modelData) {
  if (isMongoConnected) {
    await SVMModel.updateMany({ status: 'active' }, { status: 'archived' });
    const model = new SVMModel(modelData);
    return await model.save();
  }
  
  // Archive other active models in local fallback
  localData.models.forEach(m => {
    if (m.status === 'active') m.status = 'archived';
  });
  
  const newModel = {
    _id: 'model_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
    ...modelData,
    createdAt: new Date().toISOString()
  };
  localData.models.push(newModel);
  saveFallback();
  return newModel;
}

export async function deployModel(id) {
  if (isMongoConnected) {
    await SVMModel.updateMany({}, { status: 'archived' });
    return await SVMModel.findByIdAndUpdate(id, { status: 'active' }, { new: true });
  }
  
  localData.models.forEach(m => {
    m.status = m._id === id ? 'active' : 'archived';
  });
  saveFallback();
  return localData.models.find(m => m._id === id);
}

export async function deleteModel(id) {
  if (isMongoConnected) {
    return await SVMModel.findByIdAndDelete(id);
  }
  
  const initialCount = localData.models.length;
  localData.models = localData.models.filter(m => m._id !== id);
  saveFallback();
  return { deletedCount: initialCount - localData.models.length };
}
