# Models

---

## Image Classification (Fine-Tuned)

### CNNs
- [ResNet](https://huggingface.co/timm/resnet50.a1_in1k)
- [ConvNeXt](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384)
- [EfficientNet](https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k)
- [CAFormer](https://huggingface.co/timm/caformer_s18.sail_in1k_384)

### ViTs
- [ViT-B/16](https://huggingface.co/timm/vit_base_patch16_384.orig_in21k_ft_in1k)
- ViT with CNN backbone: [Google Hybrid CNN-ViT](https://huggingface.co/google/vit-hybrid-base-bit-384)

---

## Image Classification (Zero-Shot)

### OpenCLIP
- [BioCLIP](https://huggingface.co/imageomics/bioclip-2)

### timm
- TBD

### HuggingFace
- TBD

---

## MelaNet

General wrapper for tasks:
- **Classification**: Using _fine-tuned_ models classify provided images into given labels
- **Zero-Shot classification**: Using _pre-trained_ model extract features (embeddings) from provided images
- **Mixed feature extraction**: **Combine** extracted features from _fine-tuned_ model and _pre-trained_ (zero-shot) model

Additionally has utilities for embeddings and vector stores (based on [faiss](https://github.com/facebookresearch/faiss))