# Datasets

---

## Image classification
- multiclass labeled datasets

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

### MelanoMix
- created dataset from combination of __ISIC2019__ and __MILK10k__ datasets
- final splits are created iterativly
- used stratified sampling (by class label) to create splits => alleviates the problem of Random Sampling in datasets with an imbalanced-class distribution
    - keeps the distribution of classes in each of the train, validation, and test sets preserved
- Spliting procedure
    - Dataset concatenation ("single split")
    - Split dataset into __real__ train part and temporary validation/test part
    - Split the temporarily created part into __final__ validation and test parts
- interactive notebook can be found [here](./merge_datasets.ipynb)