import mongoose from 'mongoose';

const sampleSchema = new mongoose.Schema({
  imageUrl: {
    type: String,
    required: true
  },
  label: {
    type: Number, // -1 for Cat, 1 for Dog
    required: true,
    enum: [-1, 1]
  },
  features: {
    type: [Number], // 1024-dimensional feature vector
    required: true,
    validate: {
      validator: function (v) {
        return v.length === 1024;
      },
      message: props => `Feature vector must be exactly 1024 dimensions! Got ${props.value.length}`
    }
  },
  isDemo: {
    type: Boolean,
    default: false
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

const Sample = mongoose.model('Sample', sampleSchema);
export default Sample;
