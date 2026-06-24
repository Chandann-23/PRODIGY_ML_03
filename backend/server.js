import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import svmRoutes from './routes/svmRoutes.js';
import { setMongoConnected } from './utils/db.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '15mb' })); // Support base64 image uploads and features
app.use(express.urlencoded({ limit: '15mb', extended: true }));

// Database connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/svm_mlops';

mongoose.connect(MONGODB_URI, { serverSelectionTimeoutMS: 4000 })
  .then(() => {
    console.log('Successfully connected to MongoDB.');
    setMongoConnected(true);
  })
  .catch((err) => {
    console.warn('MongoDB connection failed:', err.message);
    console.log('Running backend in LOCAL JSON FILE FALLBACK mode.');
    setMongoConnected(false);
  });

// API Routes
app.use('/api', svmRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'SVM MLOps Backend running.',
    time: new Date()
  });
});

app.listen(PORT, () => {
  console.log(`Backend server running on port ${PORT}`);
});
export default app;
