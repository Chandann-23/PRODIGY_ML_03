import mongoose from 'mongoose';

const supportVectorSchema = new mongoose.Schema({
  features: {
    type: [Number],
    required: true
  },
  label: {
    type: Number,
    required: true
  },
  alpha: {
    type: Number,
    required: true
  }
});

const svmModelSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  version: {
    type: String,
    required: true
  },
  status: {
    type: String,
    enum: ['active', 'archived'],
    default: 'archived'
  },
  kernel: {
    type: String,
    enum: ['linear', 'rbf'],
    required: true
  },
  hyperparameters: {
    C: { type: Number, required: true },
    gamma: { type: Number, default: 0.1 } // relevant for RBF
  },
  weights: {
    type: [Number], // Only populated for linear kernel
    default: []
  },
  bias: {
    type: Number,
    required: true
  },
  supportVectors: [supportVectorSchema],
  metrics: {
    accuracy: { type: Number, required: true },
    precision: { type: Number, required: true },
    recall: { type: Number, required: true },
    f1: { type: Number, required: true },
    confusionMatrix: {
      tp: { type: Number, required: true }, // true dog classified as dog
      tn: { type: Number, required: true }, // true cat classified as cat
      fp: { type: Number, required: true }, // true cat classified as dog
      fn: { type: Number, required: true }  // true dog classified as cat
    },
    trainingTimeMs: { type: Number, required: true }
  },
  centroids: {
    cat: { type: [Number], default: [] },
    dog: { type: [Number], default: [] }
  },
  pcaMatrix: {
    type: [[Number]], // 2 x 1024 projection matrix for PCA visualization
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

const SVMModel = mongoose.model('SVMModel', svmModelSchema);
export default SVMModel;
