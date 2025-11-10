# 🎯 QUICK REFERENCE CARD

**Print this and keep it handy for your interview!**

---

## Project: Person Segmentation with U-Net

**Built:** November 2025 (1 week learning project)  
**Purpose:** Learn semantic segmentation for InnerGize ML internship  
**Tech:** PyTorch, U-Net, OpenCV, Albumentations

---

## ⚡ Key Numbers

| Metric | Value |
|--------|-------|
| **Architecture** | U-Net (encoder-decoder + skip connections) |
| **Parameters** | ~31 million |
| **Input Size** | 256×256×3 (RGB) |
| **Output Size** | 256×256×1 (binary mask) |
| **Encoder Blocks** | 4 (64→128→256→512 channels) |
| **Decoder Blocks** | 4 (512→256→128→64 channels) |
| **Skip Connections** | 4 (preserve spatial info) |

---

## 📊 Metrics I Implemented

### Dice Coefficient (F1 for Segmentation)
```
Formula: 2 × |A ∩ B| / (|A| + |B|)
Range: 0-1 (higher = better)
Use: Handles class imbalance well
```

### IoU (Intersection over Union)
```
Formula: |A ∩ B| / |A ∪ B|
Range: 0-1 (higher = better)
Use: Stricter than Dice, standard metric
```

### Combined Loss
```
Loss = α × BCE + (1-α) × Dice Loss
Why: BCE for pixel-level, Dice for global structure
```

---

## 🏗️ Architecture Flow

```
Input (3, 256, 256)
    ↓
[Conv-BN-ReLU] × 2 → 64 channels  ←──┐
    ↓ MaxPool                          │
[Conv-BN-ReLU] × 2 → 128 channels ←─┐ │
    ↓ MaxPool                         │ │
[Conv-BN-ReLU] × 2 → 256 channels ←┐│ │
    ↓ MaxPool                        ││ │
[Conv-BN-ReLU] × 2 → 512 channels ←┤│ │
    ↓ MaxPool                       ││ │
[Conv-BN-ReLU] × 2 → 512 (bottleneck)│ │
    ↓ Upsample                      ││ │
[Concat + Conv] → 256 channels ─────┘│ │
    ↓ Upsample                        │ │
[Concat + Conv] → 128 channels ───────┘ │
    ↓ Upsample                          │
[Concat + Conv] → 64 channels ──────────┘
    ↓ Upsample
[1×1 Conv] → 1 channel
    ↓
Output (1, 256, 256)
```

**Key:** Skip connections (→) combine encoder features with decoder

---

## 🎓 Quick Answers

### "What is U-Net?"
> Encoder-decoder architecture with skip connections. Encoder captures context 
> via downsampling, decoder enables precise localization via upsampling. Skip 
> connections preserve spatial details lost during downsampling.

### "Why U-Net for segmentation?"
> Skip connections are crucial - they combine high-resolution features from 
> encoder with semantic features from decoder. Originally designed for medical 
> imaging with limited data. Industry standard for segmentation.

### "Dice vs IoU?"
> Both measure overlap. Dice = 2×intersection/(sum of sets), IoU = intersection/union. 
> Dice is more forgiving with class imbalance (2× in numerator). IoU is stricter. 
> Medical imaging often uses Dice; computer vision often uses IoU. I implemented both.

### "How did you handle overfitting?"
> Multiple strategies: (1) Heavy data augmentation (flip, rotate, color jitter), 
> (2) Batch normalization for stability, (3) Weight decay (L2 regularization), 
> (4) Train/val split for monitoring, (5) Save best model only.

### "What's your training pipeline?"
> (1) Load images + masks with augmentation, (2) Forward pass through U-Net, 
> (3) Calculate combined loss (BCE + Dice), (4) Backprop + optimizer step, 
> (5) Validate on held-out data, (6) Track metrics, (7) Save best model.

---

## 💻 Code Highlights

### Model Creation
```python
from models import UNet
model = UNet(n_channels=3, n_classes=1, bilinear=True)
# ~31M parameters
```

### Training
```python
from utils import CombinedLoss, evaluate_metrics
criterion = CombinedLoss(alpha=0.5)  # BCE + Dice
optimizer = Adam(model.parameters(), lr=0.001)
```

### Metrics
```python
from utils import dice_coefficient, iou_score
dice = dice_coefficient(predictions, targets)
iou = iou_score(predictions, targets)
```

---

## 🔧 Data Augmentation (Albumentations)

- **Geometric:** Horizontal flip (50%), Rotation ±15° (50%)
- **Color:** Brightness/contrast ±20% (50%)
- **Blur:** Gaussian blur (30%)
- **Normalization:** ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

**Why:** Increases dataset diversity, prevents overfitting, makes model robust

---

## 📁 Project Structure (Quick)

```
models/unet.py          - U-Net implementation
utils/metrics.py        - Dice, IoU, losses
utils/dataset.py        - Data loading + augmentation
utils/visualization.py  - Plotting tools
train.py                - Training pipeline
inference.py            - Inference script
```

---

## 🎯 Relevance to InnerGize

| Requirement | My Experience |
|-------------|---------------|
| Segmentation models | ✅ U-Net implemented from scratch |
| PyTorch/TensorFlow | ✅ PyTorch training pipeline |
| Dice, IoU metrics | ✅ Both implemented and understood |
| Computer vision | ✅ OpenCV, image preprocessing |
| Medical imaging | ✅ Rapid-test-analyzer + this project |
| Data augmentation | ✅ Albumentations pipeline |

---

## 💡 What I Learned (1 Week)

✅ Implemented CNN architecture from scratch  
✅ Understanding of encoder-decoder patterns  
✅ Skip connections and why they matter  
✅ Loss function design for segmentation  
✅ Evaluation metrics (Dice, IoU)  
✅ PyTorch training loops  
✅ Data augmentation strategies  
✅ Medical imaging workflows  

---

## 🗣️ Key Phrases to Use

✅ "I built this last week specifically to learn segmentation"  
✅ "Skip connections preserve spatial information during upsampling"  
✅ "Dice coefficient handles class imbalance better than BCE alone"  
✅ "The architecture has ~31M parameters with 4 encoder-decoder stages"  
✅ "I can explain every component because I implemented it myself"  
✅ "These skills directly transfer to ear placement region detection"  

---

## ⚠️ Don't Say

❌ "I've been doing segmentation for years"  
❌ "I'm an expert in deep learning"  
❌ Anything you can't back up with code  
❌ Memorized definitions without understanding  

---

## 🎯 Interview Strategy

1. **Be Honest:** "Built this last week to learn for this role"
2. **Show Understanding:** Explain architecture, not just memorize
3. **Demonstrate Code:** Offer to walk through implementation
4. **Connect Dots:** "Skills transfer to your ear detection problem"
5. **Show Enthusiasm:** "Excited to learn more under mentorship"

---

## 📞 Emergency Reminders

- **U-Net author:** Ronneberger et al., 2015
- **Originally for:** Biomedical image segmentation
- **Key innovation:** Skip connections
- **My implementation:** PyTorch, 1,500 lines, 1 week
- **Can demo:** Training, inference, metrics
- **Combined with:** Rapid-test-analyzer (medical imaging)

---

## 🚀 Confidence Boosters

You have:
✅ Working code (can run live demo)
✅ Real understanding (built it yourself)
✅ Relevant skills (exactly what they need)
✅ Initiative (proactive learning)
✅ Medical context (rapid-test-analyzer)
✅ Honesty (transparent about timeline)

You are a **strong candidate** for an ML internship!

---

**Print this card. Keep it during interview. You've got this! 💪**

---

*Quick Commands:*
```bash
# Train
python train.py --epochs 50 --batch-size 8

# Inference
python inference.py --image test.jpg --visualize

# Test
python test_setup.py
```
