# Learning Log

Personal notes while building this project.

---

## Day 1 - November 7

Started looking into segmentation. Found U-Net paper - seems to be the standard architecture.

Key things I learned:
- Segmentation is pixel-level classification (not bounding boxes)
- U-Net has encoder-decoder structure
- Skip connections are important - they preserve spatial details

Watched a few YouTube tutorials to understand the architecture better.

---

## Day 2 - November 8

Implemented basic U-Net structure. Had some issues with:
- Getting the dimensions right (had to debug shapes a lot)
- Understanding how skip connections work (concatenate, not add!)
- Batch normalization placement

Got the model to run but need to understand the training loop better.

TODO:
- [ ] Figure out loss function (BCE alone doesn't seem right)
- [ ] Implement metrics (Dice, IoU)
- [ ] Set up data loading

---

## Day 3 - November 9

Working on training pipeline:
- Learned about Dice loss - makes sense for imbalanced segmentation
- Combined BCE + Dice loss seems to work better
- Data augmentation is tricky - need to transform both image AND mask

Issues I ran into:
- Memory errors with large batch sizes (using 4 now)
- Augmentation transforms need to be applied together
- Understanding when to use `.detach()` and `.item()`

Got first training run working! Loss is decreasing at least.

---

## Day 4 - November 10

Trying to improve results:
- Added learning rate scheduler (ReduceLROnPlateau)
- Experimenting with different augmentation strategies
- Working on visualization to see what's happening

Things I still need to understand better:
- How to know if model is overfitting
- What good Dice/IoU scores are
- How to debug when predictions look bad

Current challenges:
- Dataset is small, might need more data
- Training is slow on CPU (no GPU access right now)
- Not sure if hyperparameters are optimal

---

## Questions/Notes

**Why skip connections?**
During downsampling, spatial info is lost. Skip connections pass high-res features directly to decoder. This helps with precise localization.

**Dice vs IoU?**
Both measure overlap. Dice = 2*intersection / (sum). IoU = intersection / union.
Dice is more forgiving with imbalance. Medical imaging uses Dice a lot.

**Why combined loss?**
BCE works pixel-by-pixel. Dice looks at overall overlap. Using both gives better results than either alone.

**Data augmentation:**
- Must apply same transform to image and mask
- Albumentations makes this easier
- Common: flips, rotations, brightness changes

---

## Resources I Used

- U-Net paper: https://arxiv.org/abs/1505.04597
- PyTorch docs on custom datasets
- Various blog posts on segmentation metrics
- YouTube videos explaining encoder-decoder architectures
- Stack Overflow for debugging issues

---

## Next Steps

- [ ] Try training on better dataset
- [ ] Experiment with DeepLabV3+ (heard it's good)
- [ ] Add more augmentation techniques
- [ ] Figure out how to evaluate results better
- [ ] Maybe try different backbone architectures?

---

*This is a learning project - lots of trial and error!*
