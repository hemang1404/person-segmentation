"""
Training script for person segmentation

Usage:
    python train.py --epochs 50 --batch-size 8 --lr 0.001

Note: adjust batch-size based on your GPU memory
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json

from models import UNet
from utils import (
    create_data_loaders,
    CombinedLoss,
    evaluate_metrics,
    visualize_batch,
    plot_training_history
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # backward
        loss.backward()
        optimizer.step()
        
        # calculate metrics for monitoring
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            metrics = evaluate_metrics(preds, masks)
        
        # Update statistics
        running_loss += loss.item()
        running_dice += metrics['dice']
        running_iou += metrics['iou']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}",
            'iou': f"{metrics['iou']:.4f}"
        })
    
    # Calculate epoch averages
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_iou


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Calculate metrics
        preds = torch.sigmoid(outputs)
        metrics = evaluate_metrics(preds, masks)
        
        # Update statistics
        running_loss += loss.item()
        running_dice += metrics['dice']
        running_iou += metrics['iou']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}",
            'iou': f"{metrics['iou']:.4f}"
        })
    
    # Calculate epoch averages
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_iou


def save_checkpoint(model, optimizer, epoch, best_dice, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def train(args):
    """Main training loop"""
    
    # device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load data
    print("\nLoading dataset...")
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(device)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': []
    }
    
    # Best model tracking
    best_dice = 0.0
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_dice)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_dice, save_path)
            print(f"  ✓ New best model! Dice: {best_dice:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, best_dice, save_path)
        
        # Visualize predictions every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                images, masks = next(iter(val_loader))
                images = images.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                
                vis_path = os.path.join(args.checkpoint_dir, f'predictions_epoch_{epoch}.png')
                visualize_batch(images[:4], masks[:4], preds[:4], save_path=vis_path)
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, best_dice, final_path)
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved to {history_path}")
    
    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train Person Segmentation Model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size for training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Save parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("=" * 80)
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()
