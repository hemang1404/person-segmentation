# Project Notes: Person Segmentation

**Started:** November 2025  
**Tech:** PyTorch, U-Net, OpenCV, Albumentations  
**Why:** Want to understand how segmentation works, especially for medical imaging applications

---

## Goal

Build a system that can segment people from backgrounds. Practical uses: virtual backgrounds, photo editing, etc. But mainly learning the fundamentals of how segmentation architectures work.

---

## What I'm Learning

### 1. U-Net Architecture
- **U-Net**: Encoder-decoder architecture with skip connections
- **Why U-Net?**: Originally designed for medical imaging, perfect for precise segmentation
- **Architecture Components**:
  - Encoder: Captures context (downsampling path)
  - Decoder: Enables localization (upsampling path)
  - Skip connections: Combines high-res features with semantic information

### 2. Segmentation Basics
- Pixel-level classification - every pixel gets a label
- Different from bounding boxes (that's detection)
- This project: binary (person vs background)

### 3. Metrics - Still Understanding These
**Dice Coefficient:**
- Formula: `2 * |intersection| / |sum of both|`
- 0 to 1, higher = better
- Works better than accuracy for imbalanced data (lots of background pixels)

**IoU (Intersection over Union):**
- Formula: `|intersection| / |union|`
- Stricter than Dice
- Standard metric in CV competitions

Still figuring out when to use which one. Dice seems popular in medical imaging.

### 4. Loss Functions - Trial and Error Here
- Started with just BCE (binary cross-entropy)
- Added Dice loss after reading that it helps with class imbalance
- Using combined loss now: `alpha * BCE + (1-alpha) * Dice`
- Need to experiment more with the alpha value

### 5. Data Augmentation
Learned that for segmentation, you need to transform BOTH image and mask together!
Using Albumentations:
- Horizontal flips
- Small rotations
- Brightness/contrast changes
- Important: Same random transforms applied to image and mask

### 6. Training Stuff
- Adam optimizer (learning rate 0.001 to start)
- ReduceLROnPlateau - reduces LR when validation stops improving
- Batch norm helps training stability
- Save best model based on validation Dice (not loss)

---

## 💻 Technical Implementation

### Model Architecture (U-Net)
```
Input: RGB Image (3, 256, 256)
    ↓
Encoder Block 1: Conv-BN-ReLU → (64, 256, 256)
    ↓ MaxPool
Encoder Block 2: Conv-BN-ReLU → (128, 128, 128)
    ↓ MaxPool
Encoder Block 3: Conv-BN-ReLU → (256, 64, 64)
    ↓ MaxPool
Encoder Block 4: Conv-BN-ReLU → (512, 32, 32)
    ↓ MaxPool
Bottleneck: Conv-BN-ReLU → (512, 16, 16)
    ↓ Upsample + Skip Connection
Decoder Block 1: Conv-BN-ReLU → (256, 32, 32)
    ↓ Upsample + Skip Connection
Decoder Block 2: Conv-BN-ReLU → (128, 64, 64)
    ↓ Upsample + Skip Connection
Decoder Block 3: Conv-BN-ReLU → (64, 128, 128)
    ↓ Upsample + Skip Connection
Decoder Block 4: Conv-BN-ReLU → (64, 256, 256)
    ↓
Output: 1×1 Conv → (1, 256, 256)
```

**Total Parameters:** ~31 million

### Training Pipeline
1. **Data Loading**: Custom PyTorch Dataset with transforms
2. **Forward Pass**: Image → Model → Logits
3. **Loss Calculation**: Combined BCE + Dice Loss
4. **Backward Pass**: Gradients → Optimizer update
5. **Validation**: Check metrics on held-out data
6. **Save Best**: Keep model with highest validation Dice

### Inference Pipeline
1. **Load Image**: Read and convert to RGB
2. **Preprocess**: Resize, normalize (ImageNet stats)
3. **Forward Pass**: Model → Logits → Sigmoid → Probabilities
4. **Threshold**: Convert probabilities to binary mask (> 0.5)
5. **Post-process**: Resize to original dimensions
6. **Output**: Mask, visualization, background-removed image

---

## 📊 Project Structure

```
person-segmentation/
├── models/
│   ├── unet.py              # U-Net implementation (core architecture)
│   └── __init__.py
├── utils/
│   ├── dataset.py           # Data loading and augmentation
│   ├── metrics.py           # Dice, IoU, loss functions
│   ├── visualization.py     # Plotting and visualization
│   └── __init__.py
├── scripts/
│   └── download_dataset.py  # Dataset helper
├── train.py                 # Training script (main)
├── inference.py             # Inference script
├── examples.py              # Usage examples
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

**Total Lines of Code:** ~1,500 (well-commented)

---

## 🎓 Key Concepts Demonstrated

### Computer Vision
✓ Image preprocessing and normalization  
✓ Data augmentation strategies  
✓ Binary segmentation  
✓ Mask visualization and overlay  

### Deep Learning
✓ CNN architectures (encoder-decoder)  
✓ Skip connections (feature fusion)  
✓ Batch normalization  
✓ Training loop implementation  
✓ Loss function design  

### Software Engineering
✓ Modular code structure  
✓ Object-oriented design (Dataset, Model classes)  
✓ Command-line interfaces (argparse)  
✓ Error handling and validation  
✓ Documentation and comments  

### Machine Learning Engineering
✓ Train/validation split  
✓ Checkpoint saving and loading  
✓ Metric tracking and visualization  
✓ Hyperparameter management  
✓ Reproducibility (random seeds)  

---

## 🔧 Challenges Solved

### 1. Class Imbalance
**Problem:** Background pixels >> person pixels  
**Solution:** Combined Loss (BCE + Dice) - Dice handles imbalance better

### 2. Small Dataset
**Problem:** Limited training data  
**Solution:** Heavy data augmentation (flip, rotate, color jitter)

### 3. Memory Constraints
**Problem:** Large images, limited GPU memory  
**Solution:** Resize to 256×256, adjustable batch size

### 4. Overfitting
**Problem:** Model memorizes training data  
**Solution:** Augmentation + weight decay + validation monitoring

### 5. Training Stability
**Problem:** Loss fluctuations  
**Solution:** Batch normalization + learning rate scheduling

---

## 📈 Expected Performance

With proper dataset (5K+ images):
- **Dice Coefficient:** 0.85-0.92
- **IoU:** 0.75-0.85
- **Pixel Accuracy:** 0.95+

Training time (GPU):
- ~10-15 minutes per epoch (5K images, batch size 8)
- 50 epochs = ~8-12 hours

---

## 🚀 Potential Extensions

### Technical Improvements
1. **DeepLabV3+**: More advanced architecture
2. **Attention Mechanisms**: Focus on important regions
3. **Multi-scale Training**: Different input sizes
4. **Test-Time Augmentation**: Average multiple predictions

### Features
1. **Multi-class Segmentation**: Segment body parts
2. **Video Segmentation**: Temporal consistency
3. **Real-time Inference**: Optimize for speed (ONNX, TensorRT)
4. **Web Demo**: Gradio/Streamlit interface

### Applications
1. **Portrait Mode**: Mobile camera effects
2. **Virtual Try-on**: E-commerce applications
3. **Video Conferencing**: Background replacement
4. **Dataset Creation**: Auto-annotation tool

---

## 📝 Interview Talking Points

### "Why did you choose U-Net?"
"U-Net is the industry standard for segmentation tasks. The skip connections preserve spatial information lost during downsampling, which is crucial for accurate pixel-level predictions. It's also been proven effective in medical imaging and other domains."

### "What's the difference between Dice and IoU?"
"Both measure overlap, but Dice is more forgiving with class imbalance. Dice is the harmonic mean of precision and recall (F1 score), while IoU is stricter. For medical imaging or cases with small objects, Dice is often preferred. I implemented both to get a comprehensive view."

### "How did you handle data augmentation?"
"I used Albumentations for efficient augmentation - geometric transforms like flips and rotations for invariance, and color jittering for robustness to lighting. The key was applying the same transform to both image and mask to maintain correspondence."

### "What was the biggest challenge?"
"Balancing model capacity with overfitting. With limited data, I had to rely heavily on augmentation and regularization (weight decay, batch norm). Monitoring validation metrics closely helped me know when to stop training."

### "How does this relate to medical imaging?"
"The techniques are identical! U-Net was originally designed for medical image segmentation. Whether it's segmenting tumors, organs, or people, the core concepts (encoder-decoder, skip connections, Dice loss) remain the same. This project gave me hands-on experience with medical imaging workflows."

---

## 🎯 Relevance to InnerGize Role

### Direct Skills Match
✓ **Segmentation Models**: U-Net implementation  
✓ **PyTorch**: Full training and inference pipeline  
✓ **Evaluation Metrics**: Dice, IoU - exactly what they need  
✓ **Medical Domain**: Understanding of healthcare ML  

### Transferable Skills
✓ **Data Pipeline**: Can adapt to ear placement datasets  
✓ **Model Training**: Experience with full ML lifecycle  
✓ **Experiment Tracking**: Systematic approach to ML projects  
✓ **Documentation**: Clean, maintainable code  

### Learning Mindset
✓ Built this as a learning project (shows initiative)  
✓ Documented the process (teaching myself)  
✓ Ready to apply to new domain (ear region detection)  
✓ Genuine interest in healthcare applications  

---

**Built in:** ~1 week (November 2025)  
**Status:** Learning project, actively training and iterating  
**Next Steps:** Experiment with DeepLabV3+, try medical imaging datasets

---

*This project demonstrates my ability to learn quickly, implement complex architectures, and apply ML to practical problems. While I'm still learning, I understand the fundamentals and am ready to contribute to real-world segmentation challenges.*
