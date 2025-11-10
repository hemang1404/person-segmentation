"""
Direct download of person segmentation dataset without API keys
Uses publicly available datasets from GitHub and direct sources
"""

import os
import requests
import zipfile
from tqdm import tqdm
import shutil

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✓ Extraction complete!")

def organize_dataset(source_dir, data_dir):
    """Organize downloaded dataset into images and masks folders"""
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    # Create directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Clear existing synthetic data
    for f in os.listdir(images_dir):
        os.remove(os.path.join(images_dir, f))
    for f in os.listdir(masks_dir):
        os.remove(os.path.join(masks_dir, f))
    
    print("Looking for images and masks in downloaded data...")
    
    # Common patterns for segmentation datasets
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Walk through downloaded directory
    image_files = []
    mask_files = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_lower = file.lower()
            file_path = os.path.join(root, file)
            
            # Check if it's a mask (usually in 'mask', 'masks', 'labels', or 'annotations' folder)
            is_mask_dir = any(x in root.lower() for x in ['mask', 'label', 'annotation', 'gt', 'groundtruth'])
            
            # Or has 'mask' in filename
            is_mask_file = any(x in file_lower for x in ['mask', 'label', 'gt', 'seg'])
            
            if any(file_lower.endswith(ext) for ext in image_extensions):
                if is_mask_dir or is_mask_file:
                    mask_files.append(file_path)
                else:
                    image_files.append(file_path)
    
    print(f"Found {len(image_files)} potential images")
    print(f"Found {len(mask_files)} potential masks")
    
    # Copy files
    if image_files:
        print("Copying images...")
        for i, img_path in enumerate(tqdm(image_files)):
            ext = os.path.splitext(img_path)[1]
            shutil.copy2(img_path, os.path.join(images_dir, f'person_{i:04d}{ext}'))
    
    if mask_files:
        print("Copying masks...")
        for i, mask_path in enumerate(tqdm(mask_files)):
            ext = os.path.splitext(mask_path)[1]
            shutil.copy2(mask_path, os.path.join(masks_dir, f'person_{i:04d}{ext}'))
    
    # If images and masks are in same folder (check for pairs)
    if len(mask_files) == 0:
        print("Masks not found in separate folder. Checking for image-mask pairs...")
        # Try to match pairs by similar names
        
    final_images = len(os.listdir(images_dir))
    final_masks = len(os.listdir(masks_dir))
    
    return final_images, final_masks

def main():
    print("=" * 80)
    print("Person Segmentation Dataset - Direct Download")
    print("=" * 80)
    print()
    
    # Dataset options that don't require API keys
    datasets = {
        '1': {
            'name': 'Human Segmentation Dataset (GitHub)',
            'url': 'https://github.com/VikramShenoy97/Human-Segmentation-Dataset/archive/refs/heads/master.zip',
            'size': 'Small (~100 images)',
        },
        '2': {
            'name': 'Portrait Segmentation (Sample)',
            'url': 'https://github.com/anilsathyan7/Portrait-Segmentation/archive/refs/heads/master.zip',
            'size': 'Small (~50 samples)',
        },
        '3': {
            'name': 'COCO Sample (Manual Download)',
            'url': None,
            'size': 'Manual',
        }
    }
    
    print("Available datasets:")
    for key, dataset in datasets.items():
        if dataset['url']:
            print(f"{key}. {dataset['name']}")
            print(f"   Size: {dataset['size']}")
            print()
    
    choice = input("Select dataset (1-2) [1]: ").strip() or '1'
    
    if choice not in datasets or datasets[choice]['url'] is None:
        print("Invalid choice or manual download required.")
        return
    
    selected = datasets[choice]
    print(f"\nDownloading: {selected['name']}")
    print("This may take a few minutes...")
    print()
    
    # Download
    zip_filename = 'dataset.zip'
    temp_dir = 'temp_download'
    
    try:
        download_file(selected['url'], zip_filename)
        print("✓ Download complete!")
        print()
        
        # Extract
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        extract_zip(zip_filename, temp_dir)
        
        # Organize
        print()
        print("Organizing dataset...")
        num_images, num_masks = organize_dataset(temp_dir, 'data')
        
        # Cleanup
        os.remove(zip_filename)
        shutil.rmtree(temp_dir)
        
        print()
        print("=" * 80)
        print("Dataset Setup Complete!")
        print("=" * 80)
        print(f"Images: {num_images}")
        print(f"Masks: {num_masks}")
        print()
        
        if num_images > 0 and num_masks > 0:
            print("✓ Ready to train!")
            print()
            print("Start training with:")
            print("  python train.py --epochs 50 --batch-size 8")
        else:
            print("⚠ Warning: Dataset organization may need manual adjustment")
            print(f"Check the 'data/images' and 'data/masks' folders")
            print()
            print("The downloaded data is in a format that needs manual organization.")
            print("Please:")
            print("1. Look for image files (people photos)")
            print("2. Look for mask files (black/white segmentation)")
            print("3. Place them in data/images/ and data/masks/ respectively")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Alternative: Download manually from these sources:")
        print("1. Supervisely: https://app.supervisely.com/ecosystem/projects/supervisely-person-dataset")
        print("2. Roboflow: https://universe.roboflow.com/joseph-nelson/people-segmentation")
        print()
        print("Then organize into data/images and data/masks folders")

if __name__ == '__main__':
    main()
