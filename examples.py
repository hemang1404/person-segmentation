"""
Example usage of the person segmentation model

This script demonstrates basic usage patterns
"""

import torch
from models import UNet
from utils import dice_coefficient, iou_score

def example_model_creation():
    """Example: Create and inspect the model"""
    print("=" * 60)
    print("Example 1: Creating U-Net Model")
    print("=" * 60)
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Model info
    print(f"Model created successfully!")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Input: RGB image (B, 3, H, W)")
    print(f"Output: Binary mask (B, 1, H, W)")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print()


def example_metrics():
    """Example: Calculate segmentation metrics"""
    print("=" * 60)
    print("Example 2: Calculating Metrics")
    print("=" * 60)
    
    # Simulated predictions and targets
    predictions = torch.rand(1, 1, 256, 256)  # After sigmoid
    targets = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    # Calculate metrics
    dice = dice_coefficient(predictions, targets)
    iou = iou_score(predictions, targets)
    
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"IoU Score: {iou:.4f}")
    print()


def example_inference_flow():
    """Example: Typical inference workflow"""
    print("=" * 60)
    print("Example 3: Inference Workflow")
    print("=" * 60)
    
    # 1. Load model
    model = UNet(n_channels=3, n_classes=1)
    model.eval()
    print("✓ Model loaded")
    
    # 2. Prepare input (dummy data)
    image = torch.randn(1, 3, 256, 256)
    print("✓ Image prepared")
    
    # 3. Run inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output)
    print("✓ Inference complete")
    
    # 4. Post-process
    binary_mask = (prediction > 0.5).float()
    print("✓ Mask generated")
    
    # 5. Check coverage
    person_pixels = binary_mask.sum().item()
    total_pixels = binary_mask.numel()
    coverage = person_pixels / total_pixels * 100
    
    print(f"\nResults:")
    print(f"  Person coverage: {coverage:.2f}%")
    print(f"  Mask shape: {binary_mask.shape}")
    print()


def example_training_setup():
    """Example: How to set up training"""
    print("=" * 60)
    print("Example 4: Training Setup")
    print("=" * 60)
    
    from utils import CombinedLoss
    from torch.optim import Adam
    
    # Model
    model = UNet(n_channels=3, n_classes=1)
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    print("Training components:")
    print(f"  Model: U-Net with {model.get_num_params():,} parameters")
    print(f"  Loss: Combined BCE + Dice Loss")
    print(f"  Optimizer: Adam (lr=0.001)")
    print()
    
    # Dummy training step
    model.train()
    images = torch.randn(4, 3, 256, 256)
    masks = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    
    print(f"Sample training step:")
    print(f"  Batch size: 4")
    print(f"  Loss value: {loss.item():.4f}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("PERSON SEGMENTATION - USAGE EXAMPLES")
    print("=" * 60 + "\n")
    
    example_model_creation()
    example_metrics()
    example_inference_flow()
    example_training_setup()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python scripts/download_dataset.py")
    print("  2. Run: python train.py --epochs 5")
    print("  3. Run: python inference.py --image your_image.jpg")
    print()


if __name__ == "__main__":
    main()
