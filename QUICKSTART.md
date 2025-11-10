# Person Segmentation - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

**Option A: Use Sample Data (for testing)**
```bash
python scripts/download_dataset.py --synthetic
```

**Option B: Download Real Dataset**
```bash
# Using Kaggle API
kaggle datasets download -d tapakah68/segmentation-full-body-mads-dataset
# Unzip and organize into data/images and data/masks
```

### Step 3: Train the Model

```bash
# Quick training (5 epochs for testing)
python train.py --epochs 5 --batch-size 4

# Full training (recommended)
python train.py --epochs 50 --batch-size 8 --lr 0.001
```

**Training Output:**
- Checkpoints saved in `checkpoints/`
- Best model: `checkpoints/best_model.pth`
- Training plots: `checkpoints/training_history.png`

### Step 4: Run Inference

```bash
# Single image
python inference.py --image path/to/image.jpg --output results/ --visualize --remove-bg

# Batch processing
python inference.py --image-dir path/to/images/ --output results/ --visualize
```

**Inference Output:**
- `*_mask.png`: Binary segmentation mask
- `*_comparison.png`: Side-by-side visualization
- `*_no_background.png`: Image with transparent background

---

## 📊 Model Architecture

**U-Net Details:**
- Input: RGB image (256×256×3)
- Output: Binary mask (256×256×1)
- Parameters: ~31M
- Architecture: 4 encoder blocks + bottleneck + 4 decoder blocks

---

## 🎯 Training Tips

### For Better Results:

1. **More Data**: Use 5,000+ images for good performance
2. **Augmentation**: Already included (flip, rotate, brightness)
3. **Learning Rate**: Start with 0.001, use scheduler (included)
4. **Batch Size**: 8-16 works well (adjust for your GPU)
5. **Epochs**: 50-100 epochs recommended

### Monitor Training:

Watch the validation Dice score:
- > 0.80: Good
- > 0.90: Excellent
- < 0.70: Need more training/data

---

## 🔧 Customization

### Change Image Size:
```bash
python train.py --image-size 512  # Higher resolution
```

### Adjust Learning Rate:
```bash
python train.py --lr 0.0001  # Lower for fine-tuning
```

### Use Different Checkpoint:
```bash
python inference.py --checkpoint checkpoints/checkpoint_epoch_30.pth
```

---

## 📁 Directory Structure After Setup

```
person-segmentation/
├── checkpoints/
│   ├── best_model.pth          # Best model weights
│   ├── training_history.png    # Loss/metric plots
│   └── predictions_*.png       # Training visualizations
├── data/
│   ├── images/                 # Training images
│   └── masks/                  # Ground truth masks
├── results/
│   ├── *_mask.png             # Predicted masks
│   └── *_comparison.png       # Visualizations
└── [source files]
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory:
```bash
python train.py --batch-size 4  # Reduce batch size
```

### No GPU Available:
The code automatically detects and uses CPU if GPU is not available.

### Poor Results:
- Check if masks are binary (0 and 255)
- Ensure image-mask pairs match
- Try more epochs or data augmentation

---

## 📚 Next Steps

1. **Experiment**: Try different architectures (DeepLabV3+)
2. **Optimize**: Export to ONNX for deployment
3. **Deploy**: Create web demo with Gradio
4. **Extend**: Add multi-class segmentation

---

## 📖 Learning Resources

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Segmentation Metrics Explained](https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html)

---

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Share your results!

---

Built with ❤️ for learning semantic segmentation
