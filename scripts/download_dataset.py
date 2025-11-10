"""
Download and prepare sample dataset for person segmentation

This script helps you get started with a small sample dataset.
For production use, download larger datasets like:
- Supervisely Person Dataset
- COCO Person Segmentation
- Human Parsing Dataset
"""

import os
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def create_sample_dataset():
    """
    Create a sample dataset structure
    
    Note: This creates dummy data for demonstration.
    You should replace this with actual dataset download.
    """
    
    print("=" * 80)
    print("Dataset Setup Instructions")
    print("=" * 80)
    print()
    print("This project requires a person segmentation dataset with the following structure:")
    print()
    print("data/")
    print("  ├── images/")
    print("  │   ├── img1.jpg")
    print("  │   ├── img2.jpg")
    print("  │   └── ...")
    print("  └── masks/")
    print("      ├── img1.png")
    print("      ├── img2.png")
    print("      └── ...")
    print()
    print("=" * 80)
    print("Recommended Datasets:")
    print("=" * 80)
    print()
    print("1. Supervisely Person Dataset")
    print("   - Size: ~5,000 images")
    print("   - Link: https://supervise.ly/explore/projects/supervisely-person-dataset-23304/overview")
    print()
    print("2. COCO Person Segmentation")
    print("   - Size: Large (60K+ images)")
    print("   - Link: https://cocodataset.org/#download")
    print("   - Filter for 'person' class")
    print()
    print("3. Human Parsing Dataset (LIP)")
    print("   - Size: 50,000+ images")
    print("   - Link: http://sysu-hcp.net/lip/")
    print()
    print("4. Sample Dataset (Kaggle)")
    print("   - Search: 'person segmentation' on Kaggle")
    print("   - Example: https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset")
    print()
    print("=" * 80)
    print("Quick Start:")
    print("=" * 80)
    print()
    print("Option 1: Use Kaggle Dataset (Recommended for quick start)")
    print("  1. Install Kaggle CLI: pip install kaggle")
    print("  2. Set up Kaggle API key: https://www.kaggle.com/docs/api")
    print("  3. Download dataset:")
    print("     kaggle datasets download -d tapakah68/segmentation-full-body-mads-dataset")
    print("  4. Unzip and organize into data/images and data/masks folders")
    print()
    print("Option 2: Use Sample Images")
    print("  1. Create 'data/images' and 'data/masks' directories")
    print("  2. Add your own images and corresponding masks")
    print("  3. Masks should be binary (0=background, 255=person)")
    print()
    print("Option 3: Generate Synthetic Data (For testing)")
    print("  - Run this script with --synthetic flag")
    print("  - Creates dummy data for code testing only")
    print()
    print("=" * 80)
    
    # Ask user if they want to create directories
    response = input("\nCreate empty data directories now? (y/n): ")
    
    if response.lower() == 'y':
        # Create directories
        os.makedirs('data/images', exist_ok=True)
        os.makedirs('data/masks', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print("\n✓ Created directories:")
        print("  - data/images/")
        print("  - data/masks/")
        print("  - checkpoints/")
        print("  - results/")
        print()
        print("Next steps:")
        print("  1. Add your images to data/images/")
        print("  2. Add corresponding masks to data/masks/")
        print("  3. Run: python train.py")
    else:
        print("\nSetup cancelled. Run this script again when ready.")


def create_synthetic_data(num_samples=50):
    """
    Create synthetic dataset for testing
    WARNING: Only for code testing, not for actual training!
    """
    import numpy as np
    from PIL import Image
    
    print("\nGenerating synthetic dataset...")
    print("WARNING: This is for code testing only!")
    print()
    
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/masks', exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Generating"):
        # Create random image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Create random mask (circle in center)
        mask = np.zeros((256, 256), dtype=np.uint8)
        center = (128, 128)
        radius = np.random.randint(40, 80)
        
        y, x = np.ogrid[:256, :256]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        mask[mask_area] = 255
        
        # Save
        Image.fromarray(img).save(f'data/images/sample_{i:04d}.jpg')
        Image.fromarray(mask).save(f'data/masks/sample_{i:04d}.png')
    
    print(f"\n✓ Generated {num_samples} synthetic samples")
    print("  - Images: data/images/")
    print("  - Masks: data/masks/")
    print()
    print("You can now test the training with:")
    print("  python train.py --epochs 5 --batch-size 4")


if __name__ == "__main__":
    import sys
    
    if '--synthetic' in sys.argv:
        create_synthetic_data(num_samples=100)
    else:
        create_sample_dataset()
