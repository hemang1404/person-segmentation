"""
Segmentation Metrics Implementation

Includes:
- Dice Coefficient (F1 Score for segmentation)
- IoU (Intersection over Union / Jaccard Index)
- Pixel Accuracy
"""

import torch
import numpy as np


def dice_coefficient(preds, targets, smooth=1e-6):
    """
    Calculate Dice Coefficient (F1 score for segmentation)
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        preds: Predicted masks (B, 1, H, W) or (B, H, W)
        targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient score (0-1, higher is better)
    """
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
    return dice.item()


def iou_score(preds, targets, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union / Jaccard Index)
    
    IoU = |X ∩ Y| / |X ∪ Y|
    
    Args:
        preds: Predicted masks (B, 1, H, W) or (B, H, W)
        targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score (0-1, higher is better)
    """
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(preds, targets):
    """
    Calculate pixel-wise accuracy
    
    Args:
        preds: Predicted masks (B, 1, H, W) or (B, H, W)
        targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        
    Returns:
        Pixel accuracy (0-1, higher is better)
    """
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    correct = (preds == targets).sum()
    total = targets.numel()
    accuracy = correct / total
    
    return accuracy.item()


class DiceLoss(torch.nn.Module):
    """
    Dice Loss for training segmentation models
    Loss = 1 - Dice Coefficient
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Args:
            preds: Predictions from model (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        """
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(torch.nn.Module):
    """
    Combined BCE + Dice Loss for better training
    """
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        """
        Args:
            preds: Logits from model (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        """
        bce_loss = self.bce(preds, targets)
        
        # Apply sigmoid for dice loss
        preds_sigmoid = torch.sigmoid(preds)
        dice_loss = self.dice(preds_sigmoid, targets)
        
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


def evaluate_metrics(preds, targets, threshold=0.5):
    """
    Calculate all metrics at once
    
    Args:
        preds: Predicted probabilities (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W)
        threshold: Threshold for converting probabilities to binary
        
    Returns:
        Dictionary with all metrics
    """
    # Convert predictions to binary
    preds_binary = (preds > threshold).float()
    
    metrics = {
        'dice': dice_coefficient(preds_binary, targets),
        'iou': iou_score(preds_binary, targets),
        'pixel_accuracy': pixel_accuracy(preds_binary, targets)
    }
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    torch.manual_seed(42)
    preds = torch.rand(2, 1, 256, 256)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    metrics = evaluate_metrics(preds, targets)
    print("Test Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
