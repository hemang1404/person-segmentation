# Download Person Segmentation Dataset from Kaggle
# Run this script after setting up your Kaggle API key

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Person Segmentation Data Downloader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Kaggle CLI is installed
Write-Host "Checking Kaggle CLI..." -ForegroundColor Yellow
try {
    $kaggleVersion = kaggle --version 2>&1
    Write-Host "✓ Kaggle CLI installed: $kaggleVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Kaggle CLI not found. Installing..." -ForegroundColor Red
    pip install kaggle
    Write-Host "✓ Kaggle CLI installed" -ForegroundColor Green
}

# Check for Kaggle API key
$kaggleConfigPath = "$env:USERPROFILE\.kaggle\kaggle.json"
if (-Not (Test-Path $kaggleConfigPath)) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Kaggle API Key Not Found!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please set up your Kaggle API key:" -ForegroundColor Yellow
    Write-Host "1. Go to https://www.kaggle.com/" -ForegroundColor White
    Write-Host "2. Sign in (or create free account)" -ForegroundColor White
    Write-Host "3. Click your profile → Account → API" -ForegroundColor White
    Write-Host "4. Click 'Create New API Token'" -ForegroundColor White
    Write-Host "5. Move downloaded kaggle.json to: $kaggleConfigPath" -ForegroundColor White
    Write-Host ""
    Write-Host "Then run this script again!" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Kaggle API key found" -ForegroundColor Green
Write-Host ""

# Select dataset
Write-Host "Select a dataset to download:" -ForegroundColor Cyan
Write-Host "1. MADS Dataset (1,192 images) - RECOMMENDED" -ForegroundColor White
Write-Host "2. Portrait Matting (2,000+ images)" -ForegroundColor White
Write-Host "3. Cancel" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter choice (1-3)"

$datasetName = ""
$datasetSlug = ""

switch ($choice) {
    "1" {
        $datasetName = "MADS Segmentation Dataset"
        $datasetSlug = "tapakah68/segmentation-full-body-mads-dataset"
    }
    "2" {
        $datasetName = "Portrait Matting Dataset"
        $datasetSlug = "laurentmih/aisegmentcom-matting-human-datasets"
    }
    "3" {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "Invalid choice." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Downloading: $datasetName" -ForegroundColor Green
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
Write-Host ""

# Download
try {
    kaggle datasets download -d $datasetSlug
    $zipFile = $datasetSlug.Split('/')[-1] + ".zip"
    
    Write-Host "✓ Download complete!" -ForegroundColor Green
    Write-Host ""
    
    # Extract
    Write-Host "Extracting files..." -ForegroundColor Yellow
    $tempDir = "temp_dataset"
    
    if (Test-Path $tempDir) {
        Remove-Item $tempDir -Recurse -Force
    }
    
    Expand-Archive -Path $zipFile -DestinationPath $tempDir -Force
    Write-Host "✓ Extraction complete!" -ForegroundColor Green
    Write-Host ""
    
    # Organize
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Dataset Organization Instructions" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Dataset extracted to: $tempDir" -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Explore $tempDir to find images and masks" -ForegroundColor White
    Write-Host "2. Copy/move images to: data/images/" -ForegroundColor White
    Write-Host "3. Copy/move masks to: data/masks/" -ForegroundColor White
    Write-Host "4. Ensure image-mask pairs have matching names" -ForegroundColor White
    Write-Host ""
    Write-Host "Example structure:" -ForegroundColor Yellow
    Write-Host "  data/" -ForegroundColor White
    Write-Host "    images/" -ForegroundColor White
    Write-Host "      person001.jpg" -ForegroundColor White
    Write-Host "      person002.jpg" -ForegroundColor White
    Write-Host "    masks/" -ForegroundColor White
    Write-Host "      person001.png" -ForegroundColor White
    Write-Host "      person002.png" -ForegroundColor White
    Write-Host ""
    
    # Clean up zip
    Remove-Item $zipFile -Force
    
    Write-Host "Verify your data with:" -ForegroundColor Cyan
    Write-Host "  python -c `"import os; print(f'Images: {len(os.listdir(\"data/images\"))}'); print(f'Masks: {len(os.listdir(\"data/masks\"))}')`"" -ForegroundColor White
    Write-Host ""
    Write-Host "Then start training:" -ForegroundColor Cyan
    Write-Host "  python train.py --epochs 50 --batch-size 8" -ForegroundColor White
    Write-Host ""
    Write-Host "✓ Done!" -ForegroundColor Green
    
} catch {
    Write-Host "✗ Error downloading dataset: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible solutions:" -ForegroundColor Yellow
    Write-Host "- Check your internet connection" -ForegroundColor White
    Write-Host "- Verify Kaggle API key is correct" -ForegroundColor White
    Write-Host "- Try downloading manually from Kaggle website" -ForegroundColor White
    exit 1
}
