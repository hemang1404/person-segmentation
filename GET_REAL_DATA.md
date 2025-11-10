# 🚀 Get Real Data (No API Key Required!)

## ⚡ Fastest Option: Roboflow Universe (Recommended!)

### Step 1: Visit Roboflow
Open your browser and go to:
**https://universe.roboflow.com/joseph-nelson/people-segmentation**

### Step 2: Download
1. Click the **"Download Dataset"** button (top right)
2. You'll see download options
3. Select **"Download ZIP to Computer"**
4. Choose format: **"PNG Masks"** or **"Semantic Segmentation Masks"**
5. Click **Download**

### Step 3: Extract and Organize
```powershell
# Once downloaded, extract the zip file
Expand-Archive -Path "C:\Users\dell\Downloads\people-segmentation*.zip" -DestinationPath "C:\Users\dell\Downloads\roboflow_data"

# Clear old synthetic data
Remove-Item data\images\* -Force
Remove-Item data\masks\* -Force

# Copy new data (adjust paths based on extracted structure)
# The structure varies, so explore the extracted folder first
# Typically it will be:
# - train/images/ or train/
# - train/masks/ or train/labels/

# Copy images
Copy-Item "C:\Users\dell\Downloads\roboflow_data\train\*.*" -Destination "data\images\" -Include *.jpg,*.png -Force

# Copy masks  
Copy-Item "C:\Users\dell\Downloads\roboflow_data\train\masks\*.*" -Destination "data\masks\" -Force
```

### Step 4: Verify
```powershell
python -c "import os; print(f'Images: {len(os.listdir(\"data/images\"))}'); print(f'Masks: {len(os.listdir(\"data/masks\"))}')"
```

### Step 5: Train!
```powershell
python train.py --epochs 50 --batch-size 8
```

---

## 🎯 Alternative: Supervisely (Highest Quality!)

### Step 1: Create Free Account
1. Go to: **https://app.supervisely.com/**
2. Click **"Sign Up"** (free!)
3. Use your email or Google account

### Step 2: Download Dataset
1. Visit: **https://app.supervisely.com/ecosystem/projects/supervisely-person-dataset**
2. Click **"Run"** or **"Download"**
3. The dataset will be prepared (may take a minute)
4. Download the export (usually a .tar file)

### Step 3: Extract
```powershell
# Extract the tar file (you may need 7-Zip for Windows)
# Download 7-Zip from: https://www.7-zip.org/

# Or use PowerShell with tar (Windows 10+)
tar -xf "C:\Users\dell\Downloads\supervisely-person-dataset.tar" -C "C:\Users\dell\Downloads\supervisely_data"
```

### Step 4: Organize
```powershell
# Clear synthetic data
Remove-Item data\images\* -Force
Remove-Item data\masks\* -Force

# Supervisely structure is usually:
# - img/ (images)
# - masks_machine/ or ann/ (masks)

# Copy to your project
Copy-Item "C:\Users\dell\Downloads\supervisely_data\img\*" -Destination "data\images\" -Force
Copy-Item "C:\Users\dell\Downloads\supervisely_data\masks_machine\*" -Destination "data\masks\" -Force
```

---

## 🌐 Option 3: Manual Browser Downloads

### Google Open Images - Person Segmentation
1. Visit: **https://storage.googleapis.com/openimages/web/index.html**
2. Search for "person" in segmentation
3. Download sample images with masks

### COCO Dataset (Person class)
1. Visit: **https://cocodataset.org/#download**
2. Scroll to "2017 Train images"
3. Download: **2017 Train images [118K/18GB]** (or Val for smaller size)
4. Download: **2017 Train/Val annotations [241MB]**
5. You'll need to filter for person class and convert annotations

---

## 📦 Option 4: Smaller Datasets from GitHub

### Portrait Matting
```powershell
# Download from browser:
# https://github.com/anilsathyan7/Portrait-Segmentation

# Or use git
git clone https://github.com/anilsathyan7/Portrait-Segmentation.git
# Check the repo for data links
```

### EG3D Portraits
Various portrait datasets available on GitHub with segmentation masks.

---

## 🎬 EASIEST METHOD: Click-by-Click Guide

### Using Roboflow (5 minutes total):

1. **Open browser** → Go to `https://universe.roboflow.com/`

2. **Search** → Type "person segmentation" in search box

3. **Pick a dataset** → Examples:
   - "people-segmentation" by Joseph Nelson
   - "human-segmentation" 
   - "person-detection"

4. **Click "Download Dataset"**

5. **Select format**: "PNG" or "Semantic Segmentation"

6. **Download** → Save to Downloads folder

7. **Extract** → Right-click zip → Extract All

8. **Open PowerShell** in your project:
   ```powershell
   cd C:\Users\dell\Downloads\person-segmentation
   
   # Clear old data
   Remove-Item data\images\* -Force
   Remove-Item data\masks\* -Force
   
   # Find where images and masks are in extracted folder
   # Usually in: train/images and train/masks
   # Adjust path below based on what you see:
   
   Copy-Item "C:\Users\dell\Downloads\<dataset-name>\train\*.jpg" data\images\ -Force
   Copy-Item "C:\Users\dell\Downloads\<dataset-name>\train\*.png" data\masks\ -Force
   ```

9. **Verify**:
   ```powershell
   Get-ChildItem data\images | Measure-Object
   Get-ChildItem data\masks | Measure-Object
   ```

10. **Train**:
    ```powershell
    python train.py --epochs 50 --batch-size 8
    ```

---

## ⚡ Even Faster: Use What We Have

If you want to train RIGHT NOW without downloading:

### Option: Generate More Synthetic Data
```powershell
# Generate 1000 synthetic samples instead of 100
python scripts/download_dataset.py --synthetic --num-samples 1000
```

Then train immediately:
```powershell
python train.py --epochs 50 --batch-size 8
```

**Note:** Results won't be as good as real data, but the model will train and you can demonstrate the full pipeline!

---

## 📊 Comparison

| Method | Time | Quality | Size | Difficulty |
|--------|------|---------|------|------------|
| **Roboflow** | 5 min | ⭐⭐⭐⭐ | 500-2000 | ⭐ Easy |
| **Supervisely** | 10 min | ⭐⭐⭐⭐⭐ | 5,711 | ⭐⭐ Medium |
| **Synthetic (More)** | 1 min | ⭐⭐ | Any | ⭐ Easiest |
| **COCO** | 30 min+ | ⭐⭐⭐⭐⭐ | 60,000+ | ⭐⭐⭐⭐ Hard |

---

## 🎯 My Recommendation

**For right now (InnerGize application):**

1. **If you have 5 minutes:** → Use **Roboflow** (best balance)
2. **If you have 0 minutes:** → Keep training with synthetic data
3. **If you have 15 minutes:** → Use **Supervisely** (best quality)

**After training with synthetic data, your project shows:**
- ✅ Complete implementation
- ✅ Working training pipeline  
- ✅ Successful training run
- ✅ Inference working

**That's already impressive for an application!**

You can mention in your application:
> "Currently trained on synthetic data for demonstration. The pipeline is ready to train on real datasets like Supervisely Person Dataset or COCO (person class) for production use."

---

## 💡 Pro Tip

For the InnerGize application, what matters most is:
1. ✅ You understand segmentation (you do!)
2. ✅ You can implement U-Net (you did!)
3. ✅ You know metrics (Dice, IoU implemented!)
4. ✅ You can train a model (completed!)

The exact dataset matters less for a learning project/application.

**You're ready to apply now!** 🚀

You can always retrain with real data later if they want to see improved results during the interview process.
