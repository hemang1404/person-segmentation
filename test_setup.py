"""
Quick test script to verify the project setup

Run this to ensure everything is working correctly before training
"""

import sys
import os


def test_imports():
    """Test that all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'albumentations': 'Albumentations',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn',
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - MISSING")
            failed.append(name)
    
    print()
    
    if failed:
        print("❌ Missing packages:", ", ".join(failed))
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages installed correctly!")
        return True


def test_model():
    """Test model creation and forward pass"""
    print("\n" + "=" * 60)
    print("Testing Model...")
    print("=" * 60)
    
    try:
        import torch
        from models import UNet
        
        # Create model
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {model.get_num_params():,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        output = model(dummy_input)
        
        assert output.shape == (1, 1, 256, 256), "Output shape mismatch!"
        print(f"✓ Forward pass successful")
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {output.shape}")
        
        print("\n✅ Model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_metrics():
    """Test metric calculations"""
    print("\n" + "=" * 60)
    print("Testing Metrics...")
    print("=" * 60)
    
    try:
        import torch
        from utils import dice_coefficient, iou_score, evaluate_metrics
        
        # Create dummy data
        preds = torch.rand(1, 1, 256, 256)
        targets = torch.randint(0, 2, (1, 1, 256, 256)).float()
        
        # Calculate metrics
        dice = dice_coefficient(preds, targets)
        iou = iou_score(preds, targets)
        metrics = evaluate_metrics(preds, targets)
        
        print(f"✓ Dice coefficient: {dice:.4f}")
        print(f"✓ IoU score: {iou:.4f}")
        print(f"✓ All metrics: {metrics}")
        
        print("\n✅ Metrics test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def test_dataset():
    """Test dataset loading"""
    print("\n" + "=" * 60)
    print("Testing Dataset...")
    print("=" * 60)
    
    try:
        from utils import get_train_transform, get_val_transform
        
        # Test transforms
        train_transform = get_train_transform(256)
        val_transform = get_val_transform(256)
        
        print(f"✓ Train transform created")
        print(f"✓ Validation transform created")
        
        # Check if data directory exists
        if os.path.exists('data/images') and os.path.exists('data/masks'):
            image_count = len([f for f in os.listdir('data/images') 
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
            mask_count = len([f for f in os.listdir('data/masks') 
                            if f.endswith('.png')])
            
            print(f"✓ Found {image_count} images")
            print(f"✓ Found {mask_count} masks")
            
            if image_count == 0:
                print("\n⚠️  Warning: No images found in data/images/")
                print("   Run: python scripts/download_dataset.py")
        else:
            print("\n⚠️  Warning: Data directory not found")
            print("   Run: python scripts/download_dataset.py")
        
        print("\n✅ Dataset test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 60)
    print("Testing CUDA...")
    print("=" * 60)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA available: True")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"⚠️  CUDA not available - will use CPU")
            print(f"   Training will be slower but still functional")
        
        print("\n✅ CUDA test passed!")
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False


def test_directories():
    """Test required directories"""
    print("\n" + "=" * 60)
    print("Testing Directories...")
    print("=" * 60)
    
    required_dirs = ['checkpoints', 'results', 'data']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name:15s} exists")
        else:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✓ {dir_name:15s} created")
    
    print("\n✅ Directory test passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PERSON SEGMENTATION - SETUP VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Model", test_model()))
    results.append(("Metrics", test_metrics()))
    results.append(("Dataset", test_dataset()))
    results.append(("CUDA", test_cuda()))
    results.append(("Directories", test_directories()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to start training!")
        print("\nNext steps:")
        print("  1. Prepare dataset: python scripts/download_dataset.py")
        print("  2. Test training: python train.py --epochs 2 --batch-size 2")
        print("  3. Full training: python train.py --epochs 50 --batch-size 8")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing data: python scripts/download_dataset.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
