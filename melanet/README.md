# MelaNet - ML

---

## Datasets
### ISIC 2019
- Train
    - Includes __HAM10000__, __BCN20000__, __MSK__ datasets
        - HAM10000 ~= 10015
        - BCN20000 ~= 12413
        - MSK ~= 819
        - unknown source ~= 2084
    - multiclass classification dataset (8 classes)
    - total approx 25k images (~12 unique lesions, not counting from unknown source)
- Validation
    - approx 8.2k images
    - ~6.2k images has valid (known) ground truth

### MILK10k
- Train
    - approx 10k images (~5k unique lesions, pairs of dermatoscopic and close-up images)
    - multiclass classification dataset (11 classes)
- Validation
    - Ground truth not publicly available at time of writing (2025)

---

## Training
### Image classification
- multiclass classification task
- Tested models:
    - [Hybrid CNN-ViT](https://huggingface.co/google/vit-hybrid-base-bit-384)

---

## Testing
- Used metrics:
    - Classification:
        - Accuracy, F1, Precision, Recall
        - ROC-AUC