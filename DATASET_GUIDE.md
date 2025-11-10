# Real Person Segmentation Datasets

## Quick Access Links

### 🔥 Recommended: Easy to Download

#### 1. **Supervisely Person Dataset** (Best for Quick Start)
- **Size:** 5,711 images with masks
- **Quality:** High-quality annotations
- **Link:** https://app.supervisely.com/ecosystem/projects/supervisely-person-dataset
- **Format:** Ready to use
- **Download:**
  - Visit the link above
  - Click "Download" 
  - Extract to your project folder
  - Organize into `data/images` and `data/masks`

#### 2. **Kaggle: Segmentation Full Body (MADS Dataset)**
- **Size:** 1,192 images
- **Quality:** Good for learning
- **Link:** https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset
- **Download Method:**
  ```powershell
  # Install Kaggle CLI
  pip install kaggle
  
  # Download dataset (need Kaggle API key first)
  kaggle datasets download -d tapakah68/segmentation-full-body-mads-dataset
  
  # Unzip
  Expand-Archive -Path segmentation-full-body-mads-dataset.zip -DestinationPath temp_data
  ```

#### 3. **Kaggle: Portrait Matting**
- **Size:** 2,000+ images
- **Quality:** Professional portraits
- **Link:** https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets
- **Good for:** Background removal applications

#### 4. **Deep Fashion2 (Person Segmentation)**
- **Size:** Large dataset with person annotations
- **Link:** https://github.com/switchablenorms/DeepFashion2
- **Good for:** Fashion/retail applications

### 📊 Large Research Datasets

#### 5. **COCO Dataset (Person Class)**
- **Size:** 60,000+ person instances
- **Quality:** Research-grade
- **Link:** https://cocodataset.org/#download
- **Download:**
  ```powershell
  # Download 2017 Train images (18GB)
  Invoke-WebRequest -Uri http://images.cocodataset.org/zips/train2017.zip -OutFile train2017.zip
  
  # Download annotations (241MB)
  Invoke-WebRequest -Uri http://images.cocodataset.org/annotations/annotations_trainval2017.zip -OutFile annotations.zip
  ```
- **Note:** Need to filter for person class and convert to binary masks

#### 6. **LIP (Look Into Person)**
- **Size:** 50,000 images
- **Quality:** Multi-part segmentation (can extract person)
- **Link:** http://sysu-hcp.net/lip/
- **Good for:** Detailed person parsing

#### 7. **PASCAL-Person-Part**
- **Size:** 3,533 images
- **Quality:** Research quality
- **Link:** http://roozbehm.info/pascal-parts/pascal-parts.html
- **Good for:** Person detection with parts

---

## 🚀 Easiest Setup (Step-by-Step)

### Option A: Kaggle MADS Dataset (Recommended)

**Step 1: Get Kaggle API Key**
1. Go to https://www.kaggle.com/
2. Sign in (create account if needed - it's free!)
3. Click your profile picture → Account
4. Scroll to "API" section → Click "Create New API Token"
5. This downloads `kaggle.json`
6. Move it to: `C:\Users\dell\.kaggle\kaggle.json`

**Step 2: Install Kaggle CLI**
```powershell
pip install kaggle
```

**Step 3: Download Dataset**
```powershell
cd C:\Users\dell\Downloads\person-segmentation

# Download
kaggle datasets download -d tapakah68/segmentation-full-body-mads-dataset

# Unzip
Expand-Archive -Path segmentation-full-body-mads-dataset.zip -DestinationPath temp_mads

# The dataset structure varies, check and organize:
# Move images to data/images/
# Move masks to data/masks/
```

**Step 4: Verify**
```powershell
python -c "import os; print(f'Images: {len(os.listdir(\"data/images\"))}'); print(f'Masks: {len(os.listdir(\"data/masks\"))}')"
```

---

### Option B: Direct Download (No API needed)

**1. Google Open Images (Person Segmentation)**
- Search: "open images person segmentation dataset download"
- Direct links available for subsets

**2. Sample Datasets from GitHub**
```powershell
# Example: Small person dataset
git clone https://github.com/VikramShenoy97/Human-Segmentation-Dataset.git
# Copy images and masks to your data folder
```

**3. ADE20K (Has Person Class)**
- Link: http://groups.csail.mit.edu/vision/datasets/ADE20K/
- Download scene parsing dataset
- Filter for person class

---

## 📥 Quick Download Script

Save this as `download_real_data.ps1`:

```powershell
# Download MADS dataset from Kaggle
Write-Host "Downloading person segmentation dataset..."

# Check if Kaggle is installed
try {
    kaggle --version
    Write-Host "Kaggle CLI found!"
} catch {
    Write-Host "Installing Kaggle CLI..."
    pip install kaggle
}

# Download dataset
kaggle datasets download -d tapakah68/segmentation-full-body-mads-dataset

# Unzip
Write-Host "Extracting..."
Expand-Archive -Path segmentation-full-body-mads-dataset.zip -DestinationPath temp_data -Force

Write-Host "Done! Check temp_data folder and organize into data/images and data/masks"
```

Run with:
```powershell
powershell -ExecutionPolicy Bypass -File download_real_data.ps1
```

---

## 🔄 Alternative: Use Roboflow

**Roboflow** has ready-to-use datasets:
1. Go to https://universe.roboflow.com/
2. Search for "person segmentation"
3. Many free datasets available
4. Download in your preferred format
5. Example: https://universe.roboflow.com/joseph-nelson/people-segmentation

---

## 📊 Dataset Comparison

| Dataset | Size | Quality | Download | Best For |
|---------|------|---------|----------|----------|
| MADS (Kaggle) | 1,192 | Good | Easy | Quick start |
| Supervisely | 5,711 | High | Easy | Best results |
| COCO | 60,000+ | Research | Medium | Large scale |
| LIP | 50,000 | High | Medium | Detailed parsing |
| Roboflow | Varies | Good | Very Easy | Quick experiments |

---

## ⚡ Fastest Path to Training

**10-Minute Setup:**

1. **Go to Supervisely:**
   - https://app.supervisely.com/ecosystem/projects/supervisely-person-dataset
   - Download directly (no API needed)

2. **Or use Roboflow:**
   - https://universe.roboflow.com/joseph-nelson/people-segmentation
   - Export → Download
   - Choose "PNG masks" format

3. **Organize files:**
   ```powershell
   # Put all images in data/images/
   # Put all masks in data/masks/
   # Ensure matching filenames
   ```

4. **Start training:**
   ```powershell
   python train.py --epochs 50 --batch-size 8
   ```

---

## 🎯 What I Recommend

**For this project (InnerGize application):**

Use **Kaggle MADS dataset** (1,192 images):
- ✅ Quick to download
- ✅ Good quality
- ✅ Enough for learning
- ✅ Shows you can work with real data
- ✅ Reasonable training time (few hours)

**Setup time:** 15 minutes  
**Training time:** 3-4 hours on CPU, ~30 min on GPU

---

## 📝 After Download

Once you have real data:

```powershell
# Remove synthetic data
Remove-Item data/images/* -Force
Remove-Item data/masks/* -Force

# Add real data to data/images and data/masks

# Verify
python -c "import os; print(f'Images: {len(os.listdir(\"data/images\"))}'); print(f'Masks: {len(os.listdir(\"data/masks\"))}')"

# Train with real data
python train.py --epochs 50 --batch-size 8

# Expected results with real data:
# Dice: 0.85-0.92 (vs 0.36 with synthetic)
# IoU: 0.75-0.85 (vs 0.22 with synthetic)
```

---

## 🆘 Troubleshooting

**Problem:** "Kaggle API not found"
- Solution: Install with `pip install kaggle`

**Problem:** "401 Unauthorized"
- Solution: Set up `kaggle.json` in `C:\Users\dell\.kaggle\`

**Problem:** "Images and masks don't match"
- Solution: Ensure filenames match (e.g., `img001.jpg` → `img001.png`)

**Problem:** "Dataset too large for my computer"
- Solution: Use MADS dataset (smaller, 1,192 images)

---

## 💡 Pro Tips

1. **Start small:** Use 1,000-5,000 images first
2. **Check quality:** Manually verify a few image-mask pairs
3. **Augmentation helps:** Our pipeline already has strong augmentation
4. **GPU recommended:** Training 50 epochs on 5K images = 8-12 hours (CPU) vs 1-2 hours (GPU)

---

**Ready to download?** Start with Kaggle MADS - it's the easiest!
