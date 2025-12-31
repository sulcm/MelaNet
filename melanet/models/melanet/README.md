# MelaNet

Sub-module to wrap **MelaNet** models and run inference.

---

## MelaNet Wrapper

Wrapper class for MelaNet inference.
Supported modes:
- Fine-tune classifier
- Feature extraction
- Zero-Shot feature extraction
- Combination of fine-tuned and zero-shot feature extraction

---

## Zero-Shot Models

Wrapper in zero [shot feature extraction](./zero_shot/zero_shot_wrapper.py).
Supported backends are:
- HuggingFace
- `timm`
- OpenCLIP

---

## Vector Stores

Wrappers for FAISS vector indexes.

---

## Adapters

To transform differntly sized embeddings from models there adapters such as:
- PCA: Simple non-parametric projection / extraction into lower dimensions
- Linear projection: Learnable single layer linear projection
- MLP Fusion: Fuse 2 embeddings of any sizes into single representation (must be trained)
Adapter [implementations](./adapters/).