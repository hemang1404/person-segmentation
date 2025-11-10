"""
Visualization utilities for segmentation results
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        image: Normalized image tensor (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized image as numpy array (H, W, C)
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Denormalize
    image = image.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    return image


def visualize_prediction(image, mask, prediction, save_path=None):
    """
    Visualize original image, ground truth mask, and prediction side by side
    
    Args:
        image: Input image tensor (C, H, W)
        mask: Ground truth mask (1, H, W)
        prediction: Predicted mask (1, H, W)
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Denormalize image
    img = denormalize_image(image)
    
    # Convert tensors to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().squeeze().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().squeeze().numpy()
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Plot prediction
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # Plot overlay
    overlay = img.copy()
    prediction_colored = np.zeros_like(img)
    prediction_colored[:, :, 1] = prediction  # Green channel
    overlay = cv2.addWeighted(overlay, 0.7, prediction_colored, 0.3, 0)
    axes[3].imshow(overlay)
    axes[3].set_title('Prediction Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch(images, masks, predictions, num_samples=4, save_path=None):
    """
    Visualize a batch of predictions
    
    Args:
        images: Batch of images (B, C, H, W)
        masks: Batch of masks (B, 1, H, W)
        predictions: Batch of predictions (B, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Denormalize image
        img = denormalize_image(images[i])
        
        # Convert to numpy
        mask = masks[i].cpu().squeeze().numpy()
        pred = predictions[i].cpu().squeeze().numpy()
        
        # Plot image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i+1}: Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Sample {i+1}: Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Sample {i+1}: Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_dice', 'val_dice', etc.
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Dice coefficient
    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[1].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[1].set_title('Dice Coefficient over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot IoU
    axes[2].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[2].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[2].set_title('IoU over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def remove_background(image, mask, background_color=(0, 255, 0)):
    """
    Remove background from image using predicted mask
    
    Args:
        image: Input image (H, W, C) or (C, H, W)
        mask: Binary mask (H, W) or (1, H, W)
        background_color: RGB color for background or None for transparent
        
    Returns:
        Image with background removed
    """
    # Convert to numpy if tensor
    if isinstance(image, torch.Tensor):
        image = denormalize_image(image)
        image = (image * 255).astype(np.uint8)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().squeeze().numpy()
    
    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)
    
    # Create 3-channel mask
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    
    if background_color is None:
        # Create transparent background (RGBA)
        result = np.dstack([image, mask * 255])
    else:
        # Replace background with color
        background = np.full_like(image, background_color)
        result = image * mask_3ch + background * (1 - mask_3ch)
    
    return result


if __name__ == "__main__":
    # Test visualization
    print("Visualization utilities loaded successfully")
