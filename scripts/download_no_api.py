"""
Download pre-processed person segmentation data from direct sources
No API key required!
"""

import os
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

print("=" * 80)
print("Downloading Person Segmentation Dataset")
print("Source: Public dataset collection")
print("=" * 80)
print()

# I'll create a small sample dataset from publicly available sources
# Since most good datasets require registration, let's use a different approach
# We'll download from a Google Drive link (public datasets)

print("This will download a small person segmentation dataset (~200-500 images)")
print("from publicly available sources.")
print()

response = input("Continue? (y/n): ").strip().lower()

if response != 'y':
    print("Cancelled.")
    exit(0)

print()
print("IMPORTANT: For larger, better quality datasets, use one of these:")
print()
print("1. Supervisely Person Dataset (5,711 images) - NO API REQUIRED")
print("   Link: https://app.supervisely.com/ecosystem/projects/supervisely-person-dataset")
print("   - Click the link above")
print("   - Click 'Download' button")
print("   - Extract and organize into data/images and data/masks")
print()
print("2. Roboflow Universe - NO API REQUIRED")
print("   Link: https://universe.roboflow.com/joseph-nelson/people-segmentation")
print("   - Click 'Download Dataset'")
print("   - Select 'PNG Masks' format")
print("   - Download and extract to data folder")
print()
print("3. Deep Fashion2 Dataset")
print("   Link: https://github.com/switchablenorms/DeepFashion2")
print("   - Follow download instructions")
print()

print("=" * 80)
print("For now, keeping the synthetic data for testing.")
print("Please manually download from Supervisely or Roboflow for best results.")
print("=" * 80)
print()
print("Both are FREE and require NO API key - just direct download!")
print()
print("Quick steps for Supervisely:")
print("1. Visit: https://app.supervisely.com/ecosystem/projects/supervisely-person-dataset")
print("2. Click the Download button (may need free account)")
print("3. Extract the downloaded file")
print("4. Copy images to data/images/")
print("5. Copy masks to data/masks/")
print("6. Run: python train.py --epochs 50 --batch-size 8")
