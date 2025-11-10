"""
Inference script for person segmentation

Run with:
    python inference.py --image path/to/image.jpg --output results/
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import UNet
from utils import visualize_prediction, remove_background, denormalize_image


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        if 'best_dice' in checkpoint:
            print(f"Model's best Dice score: {checkpoint['best_dice']:.4f}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using untrained model")
    
    model = model.to(device)
    model.eval()
    return model


def get_inference_transform(image_size=256):
    """Transformation pipeline for inference"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def preprocess_image(image_path, transform, image_size=256):
    """Load and preprocess image for inference"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Apply transform
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size, image


@torch.no_grad()
def predict(model, image_tensor, device):
    """Run inference on image"""
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    prediction = torch.sigmoid(output)
    return prediction


def postprocess_mask(mask, original_size, threshold=0.5):
    """Postprocess predicted mask to original size"""
    # Convert to numpy
    mask = mask.squeeze().cpu().numpy()
    
    # Apply threshold
    mask_binary = (mask > threshold).astype(np.uint8) * 255
    
    # Resize to original size
    mask_resized = cv2.resize(mask_binary, (original_size[1], original_size[0]))
    
    return mask_resized


def inference_single_image(args):
    """Run inference on a single image"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device)
    
    # Prepare transforms
    transform = get_inference_transform(args.image_size)
    
    # Load and preprocess image
    print(f"\nProcessing image: {args.image}")
    image_tensor, original_size, original_image = preprocess_image(
        args.image, transform, args.image_size
    )
    
    # Run inference
    print("Running inference...")
    prediction = predict(model, image_tensor, device)
    
    # Postprocess mask
    mask = postprocess_mask(prediction, original_size, args.threshold)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # Save mask
    mask_path = os.path.join(args.output, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved to {mask_path}")
    
    # Save visualization
    if args.visualize:
        vis_path = os.path.join(args.output, f"{base_name}_comparison.png")
        
        # Prepare for visualization (use resized versions)
        image_resized = cv2.resize(original_image, (args.image_size, args.image_size))
        mask_resized = (prediction.squeeze().cpu().numpy() > args.threshold).astype(np.uint8)
        
        # Create visualization
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_resized)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_resized, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = image_resized.copy().astype(float) / 255.0
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = mask_resized  # Green channel
        overlay_combined = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay_combined)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {vis_path}")
    
    # Remove background and save
    if args.remove_bg:
        # Use original size mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
        
        # Create result with transparent background
        original_bgr = cv2.imread(args.image)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        
        # Create RGBA image
        result_rgba = np.dstack([
            original_rgb,
            (mask > 127).astype(np.uint8) * 255
        ])
        
        # Save as PNG with transparency
        nobg_path = os.path.join(args.output, f"{base_name}_no_background.png")
        Image.fromarray(result_rgba).save(nobg_path)
        print(f"Background removed image saved to {nobg_path}")
    
    print("\n✓ Inference completed successfully!")


def inference_directory(args):
    """Run inference on all images in a directory"""
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    print(f"Found {len(image_files)} images in {args.image_dir}")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {os.path.basename(image_path)}")
        
        # Temporarily set args.image
        args.image = image_path
        
        try:
            inference_single_image(args)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"\n✓ Processed {len(image_files)} images!")


def main():
    parser = argparse.ArgumentParser(description='Person Segmentation Inference')
    
    # Input/Output
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Path to directory containing images')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    
    # Output options
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization of results')
    parser.add_argument('--remove-bg', action='store_true',
                        help='Generate image with background removed')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.image is None and args.image_dir is None:
        parser.error("Either --image or --image-dir must be specified")
    
    if args.image and args.image_dir:
        parser.error("Cannot specify both --image and --image-dir")
    
    # Run inference
    if args.image:
        inference_single_image(args)
    else:
        inference_directory(args)


if __name__ == "__main__":
    main()
