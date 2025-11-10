# Person Segmentation for Background Removal

**Status:** Learning project (started Nov 2025)  
**Goal:** Understanding semantic segmentation and U-Net architecture

Implementing U-Net for semantic segmentation in PyTorch. Started this project to learn about segmentation architectures and how they work for real-world computer vision tasks.

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

Learning semantic segmentation by building a person detection system. The goal is to understand encoder-decoder architectures and how they do pixel-level classification.

**What I'm exploring:**
- How U-Net architecture works (encoder-decoder + skip connections)
- Implementing segmentation metrics (Dice, IoU) 
- Training pipelines in PyTorch
- Real use cases: virtual backgrounds, photo editing, AR applications

## What's implemented

- U-Net architecture (encoder-decoder with skip connections)
- Training pipeline with Dice and IoU metrics
- Data augmentation using Albumentations
- Inference script for testing on images
- Visualization tools to see predictions

Still learning and improving this - feedback welcome!

## Architecture Notes

U-Net has an encoder path that downsamples (captures context) and a decoder path that upsamples (precise localization). The skip connections are the key - they pass features from encoder to decoder to preserve spatial details.

```
Input Image (3×H×W)
    ↓
[Encoder] → Skip Connections
    ↓              ↓
[Bottleneck]       ↓
    ↓              ↓
[Decoder] ← ← ← ← ←
    ↓
Output Mask (1×H×W)
```

Reading the original paper helped a lot: [U-Net Paper](https://arxiv.org/abs/1505.04597)

## Metrics

Implementing two main metrics:
- **Dice Coefficient**: 2 * (intersection) / (sum of both). Good for imbalanced classes
- **IoU**: intersection / union. Stricter than Dice
- **Pixel Accuracy**: Just counts correct pixels (less useful for segmentation)

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended) or CPU
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/hemang1404/person-segmentation.git
cd person-segmentation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
```bash
python scripts/download_dataset.py
```

### Training

```bash
# Quick test run
python train.py --epochs 5 --batch-size 4

# Longer training
python train.py --epochs 50 --batch-size 8 --lr 0.001
```

Note: Adjust batch-size based on your GPU memory. I use 4 on my setup.

### Inference

```bash
python inference.py --image path/to/image.jpg --output results/
```

**This will generate:**
- Original image
- Predicted mask
- Background removed (transparent PNG)
- Side-by-side comparison

## Project Structure

```
person-segmentation/
├── models/
│   └── unet.py              # U-Net implementation
├── utils/
│   ├── dataset.py           # Data loading + augmentation
│   ├── metrics.py           # Dice, IoU, loss functions
│   └── visualization.py     # Plotting tools
├── train.py                 # Training script
├── inference.py             # Inference on images
├── LEARNING_LOG.md          # My notes while building this
└── requirements.txt
```

## Technical Details

### Model
- Input: 256×256 RGB images
- Output: 256×256 binary masks
- 4 encoder blocks, bottleneck, 4 decoder blocks
- About 31M parameters (checked using `model.get_num_params()`)

### Training Setup
- Loss: Combining BCE and Dice loss (Dice helps with class imbalance)
- Optimizer: Adam with weight decay
- Augmentation: flips, rotations, brightness changes (using Albumentations)
- LR scheduler: ReduceLROnPlateau (reduces LR when val loss plateaus)

### Data
Working with person segmentation datasets. Need to organize as:
```
data/
  images/  # input images
  masks/   # corresponding binary masks
```

## Results

Still training and experimenting with different hyperparameters. Will update with metrics once I get good results.

Working on:
- [ ] Getting a good dataset (trying different sources)
- [ ] Training for enough epochs
- [ ] Tuning hyperparameters
- [ ] Adding sample predictions

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **Albumentations**: Advanced data augmentation
- **OpenCV**: Image processing
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization
- **Pillow**: Image I/O

## 📚 Learning Resources

This project was built by studying:
- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Ronneberger et al.
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Albumentations Documentation](https://albumentations.ai/)

## Learning Journey

Things I figured out while building this:
- How skip connections actually work (they're crucial for preserving spatial info!)
- Why Dice loss is better than just BCE for segmentation
- PyTorch DataLoaders and custom Dataset classes
- Augmentation is different for segmentation (need to transform image AND mask together)
- Debugging training - watching metrics, visualizing predictions

Still learning about optimization and how to get better results with limited data.

## TODO / Ideas

- [ ] Try DeepLabV3+ (heard it's better than U-Net for some cases)
- [ ] Add TensorBoard for tracking experiments
- [ ] Test different loss functions
- [ ] Maybe make a simple web interface with Gradio
- [ ] Video segmentation (frame-by-frame for now)
- [ ] Better data augmentation strategies

## 📝 License

MIT License - feel free to use this project for learning!

## References

- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Ronneberger et al., 2015
- PyTorch docs and tutorials
- Various blog posts and GitHub repos for understanding segmentation

Check `LEARNING_LOG.md` for my notes and questions while building this.

## Contact

Hemang Sharma - [@hemang1404](https://github.com/hemang1404)

---

*Learning project - started November 2025. Still working on improving results!*
