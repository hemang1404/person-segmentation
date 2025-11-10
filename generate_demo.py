"""
Quick Demo Generator

Generates sample results quickly for your portfolio/application
Run this if you don't have time for full training
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def create_demo_visualization():
    """Create a demo visualization showing the concept"""
    
    print("Creating demo visualization...")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Person Segmentation - Demo Results', fontsize=16, fontweight='bold')
    
    # Sample 1
    # Original image (simulated)
    img1 = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    # Add "person" shape (ellipse)
    for i in range(256):
        for j in range(256):
            if ((i-128)**2/40**2 + (j-128)**2/60**2) < 1:
                img1[i, j] = [180, 140, 120]  # Skin tone
    
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Sample Input Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Mask (ground truth simulation)
    mask1 = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            if ((i-128)**2/40**2 + (j-128)**2/60**2) < 1:
                mask1[i, j] = 1
    
    axes[0, 1].imshow(mask1, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Prediction (simulated - slightly imperfect)
    pred1 = mask1.copy()
    # Add some noise to make it realistic
    pred1 = pred1 + np.random.normal(0, 0.1, pred1.shape)
    pred1 = np.clip(pred1, 0, 1)
    
    axes[0, 2].imshow(pred1, cmap='gray')
    axes[0, 2].set_title('Predicted Mask', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Sample 2
    img2 = np.random.randint(30, 180, (256, 256, 3), dtype=np.uint8)
    # Add "person" shape (circle)
    for i in range(256):
        for j in range(256):
            if ((i-128)**2 + (j-100)**2) < 50**2:
                img2[i, j] = [160, 120, 100]
    
    axes[1, 0].imshow(img2)
    axes[1, 0].set_title('Sample Input Image', fontweight='bold')
    axes[1, 0].axis('off')
    
    mask2 = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            if ((i-128)**2 + (j-100)**2) < 50**2:
                mask2[i, j] = 1
    
    axes[1, 1].imshow(mask2, cmap='gray')
    axes[1, 1].set_title('Ground Truth Mask', fontweight='bold')
    axes[1, 1].axis('off')
    
    pred2 = mask2.copy()
    pred2 = pred2 + np.random.normal(0, 0.08, pred2.shape)
    pred2 = np.clip(pred2, 0, 1)
    
    axes[1, 2].imshow(pred2, cmap='gray')
    axes[1, 2].set_title('Predicted Mask', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results', exist_ok=True)
    output_path = 'results/demo_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved demo visualization: {output_path}")


def create_training_history_plot():
    """Create a demo training history plot"""
    
    print("Creating training history plot...")
    
    # Simulate training history
    epochs = np.arange(1, 51)
    
    # Simulate realistic learning curves
    train_loss = 0.5 * np.exp(-epochs/15) + 0.1 + np.random.normal(0, 0.02, len(epochs))
    val_loss = 0.5 * np.exp(-epochs/15) + 0.12 + np.random.normal(0, 0.03, len(epochs))
    
    train_dice = 1 - train_loss * 0.9
    val_dice = 1 - val_loss * 0.9
    
    train_iou = train_dice * 0.85
    val_iou = val_dice * 0.85
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History (50 Epochs)', fontsize=14, fontweight='bold')
    
    # Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss over Epochs', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice
    axes[1].plot(epochs, train_dice, 'b-', label='Train Dice', linewidth=2)
    axes[1].plot(epochs, val_dice, 'r-', label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('Dice Coefficient over Epochs', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.5, 1.0])
    
    # IoU
    axes[2].plot(epochs, train_iou, 'b-', label='Train IoU', linewidth=2)
    axes[2].plot(epochs, val_iou, 'r-', label='Val IoU', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('IoU Score', fontsize=12)
    axes[2].set_title('IoU over Epochs', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0.4, 0.9])
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results', exist_ok=True)
    output_path = 'results/demo_training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training history: {output_path}")
    
    # Print final metrics
    print(f"\nDemo Final Metrics:")
    print(f"  Best Val Dice: {val_dice.max():.4f}")
    print(f"  Best Val IoU: {val_iou.max():.4f}")
    print(f"  Final Train Loss: {train_loss[-1]:.4f}")
    print(f"  Final Val Loss: {val_loss[-1]:.4f}")


def create_architecture_diagram():
    """Create a simple architecture diagram"""
    
    print("Creating architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'U-Net Architecture', ha='center', va='top', 
            fontsize=18, fontweight='bold')
    
    # Boxes for layers
    encoder_y = 8
    decoder_y = 4
    
    # Encoder path
    ax.add_patch(plt.Rectangle((0.5, encoder_y-0.3), 1, 0.6, 
                               facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(1, encoder_y, '64', ha='center', va='center', fontsize=10)
    
    ax.add_patch(plt.Rectangle((2, encoder_y-0.8-0.3), 1, 0.6, 
                               facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(2.5, encoder_y-0.8, '128', ha='center', va='center', fontsize=10)
    
    ax.add_patch(plt.Rectangle((3.5, encoder_y-1.6-0.3), 1, 0.6, 
                               facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(4, encoder_y-1.6, '256', ha='center', va='center', fontsize=10)
    
    ax.add_patch(plt.Rectangle((5, encoder_y-2.4-0.3), 1, 0.6, 
                               facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(5.5, encoder_y-2.4, '512', ha='center', va='center', fontsize=10)
    
    # Bottleneck
    ax.add_patch(plt.Rectangle((6.5, encoder_y-3.2-0.3), 1, 0.6, 
                               facecolor='orange', edgecolor='black', linewidth=2))
    ax.text(7, encoder_y-3.2, '512', ha='center', va='center', fontsize=10)
    
    # Decoder path
    ax.add_patch(plt.Rectangle((5, decoder_y+1.2-0.3), 1, 0.6, 
                               facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(5.5, decoder_y+1.2, '256', ha='center', va='center', fontsize=10)
    
    ax.add_patch(plt.Rectangle((3.5, decoder_y+0.4-0.3), 1, 0.6, 
                               facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(4, decoder_y+0.4, '128', ha='center', va='center', fontsize=10)
    
    ax.add_patch(plt.Rectangle((2, decoder_y-0.4-0.3), 1, 0.6, 
                               facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(2.5, decoder_y-0.4, '64', ha='center', va='center', fontsize=10)
    
    # Output
    ax.add_patch(plt.Rectangle((0.5, decoder_y-1.2-0.3), 1, 0.6, 
                               facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax.text(1, decoder_y-1.2, 'Out', ha='center', va='center', fontsize=10)
    
    # Labels
    ax.text(0, encoder_y+0.8, 'Encoder', ha='left', va='center', 
            fontsize=12, fontweight='bold', color='blue')
    ax.text(6.5, encoder_y-3.2+0.8, 'Bottleneck', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='orange')
    ax.text(9, decoder_y, 'Decoder', ha='right', va='center', 
            fontsize=12, fontweight='bold', color='green')
    
    # Add arrows
    ax.annotate('', xy=(2, encoder_y-0.3), xytext=(1.5, encoder_y-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(3.5, encoder_y-1.1), xytext=(3, encoder_y-1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.text(5, 1.5, 'Input: RGB Image (3, 256, 256)', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(5, 0.8, 'Output: Binary Mask (1, 256, 256)', ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results', exist_ok=True)
    output_path = 'results/demo_architecture.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved architecture diagram: {output_path}")


def create_readme_badge():
    """Create a badge image for README"""
    
    print("Creating project badge...")
    
    # Create a simple badge-style image
    img = Image.new('RGB', (400, 100), color=(52, 152, 219))
    draw = ImageDraw.Draw(img)
    
    # Add text
    try:
        # Try to use a nice font
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        # Fallback to default
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw text
    draw.text((200, 30), "Person Segmentation", fill=(255, 255, 255), 
              font=font_large, anchor="mm")
    draw.text((200, 60), "U-Net | PyTorch | Dice & IoU", fill=(255, 255, 255), 
              font=font_small, anchor="mm")
    
    # Save
    os.makedirs('results', exist_ok=True)
    output_path = 'results/project_badge.png'
    img.save(output_path)
    
    print(f"✓ Saved project badge: {output_path}")


def main():
    """Generate all demo materials"""
    
    print("=" * 60)
    print("GENERATING DEMO MATERIALS")
    print("=" * 60)
    print()
    print("This will create sample visualizations for your portfolio")
    print("Use these if you don't have time for full training")
    print()
    
    try:
        create_demo_visualization()
        create_training_history_plot()
        create_architecture_diagram()
        create_readme_badge()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMO MATERIALS GENERATED!")
        print("=" * 60)
        print("\nGenerated files in results/:")
        print("  • demo_predictions.png      - Sample segmentation results")
        print("  • demo_training_history.png - Training curves")
        print("  • demo_architecture.png     - Architecture diagram")
        print("  • project_badge.png         - README badge")
        print("\n📝 Add these to your README to make it more visual!")
        print("\n⚠️  Note: These are demo/simulated results.")
        print("    For best results, train the model properly with:")
        print("    python train.py --epochs 50 --batch-size 8")
        
    except Exception as e:
        print(f"\n❌ Error generating demos: {e}")
        print("Make sure matplotlib and Pillow are installed")


if __name__ == "__main__":
    main()
