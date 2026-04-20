# Models

---

## Image Classification (Fine-Tuned)

### CNNs
- [ResNet](https://huggingface.co/timm/resnet50.a1_in1k)
- [ConvNeXt](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384)
- [EfficientNet](https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k)
- [CAFormer](https://huggingface.co/timm/caformer_s18.sail_in1k_384)

### Transformers
- [ViT-B/16](https://huggingface.co/timm/vit_base_patch16_384.orig_in21k_ft_in1k)
- ViT with CNN backbone: [Google Hybrid CNN-ViT](https://huggingface.co/google/vit-hybrid-base-bit-384)
- [Swin Transformer V2](https://huggingface.co/timm/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k)

---

## Feature extraction

### Using Fine-Tuned Model
- Same as [previous section](#image-classification-fine-tuned)

### Zero-Shot
- [DermLIP ViT-B/16](https://huggingface.co/redlessone/DermLIP_ViT-B-16)
- [DINOv3 ViT-H+/16](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3.lvd1689m)
- [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2)
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

---

## Adapters

### Linear
- Creates simple one linear layer network with some additional options
- [Implementation](./melanet/adapters/linear.py)

### Fusion
- Create fusion network that takes list of input tensors (different feature encoders) and produces single representation
- [Implementation](./melanet/adapters/fusion.py) supports different approches with configurable options:
    - Concatenation-based fusion (`use_attn=False`)
    - Attention-based fusion (`use_attn=True`)

### PCA
- Simple Principal Component Analysis applied on extracted features
- [Implementation](./melanet/adapters/pca.py)

---

## MelaNet

General wrapper for tasks:
- **Classification**: Using _fine-tuned_ models classify provided images into given labels
- **Zero-Shot classification**: Using _pre-trained_ model extract features (embeddings) from provided images
- **Mixed feature extraction**: **Combine** extracted features from _fine-tuned_ model and _pre-trained_ (zero-shot) model
- **Feature extraction with adapters**: Combine and/or adapt extracted features from _fine-tuned_ model and _pre-trained_ (zero-shot) model into logits (classification) or embeddings

Additionally has utilities for embeddings and vector stores (based on [faiss](https://github.com/facebookresearch/faiss))