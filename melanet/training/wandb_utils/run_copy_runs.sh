SRC_ENTITY="sulcm"
DST_ENTITY="sulcm"

# ======================================================================================================================================================
# SRC_PROJECTS_BASE=("MelaNet-ResNet" "MelaNet-CAFormer" "MelaNet-ViT" "MelaNet-EfficientNet" "MelaNet-ConvNeXt" "MelaNet-Swin" "MelaNet-GoogleHybridViT")


# MILK10k
# SRC_PROJECTS_SUFFIX=("Grid" "SeeSaw-Grid" "Fine-Grid")
# DST_PROJECT="MelaNet-MILK10k-Sweeps"

# MelanoMix
# SRC_PROJECTS_SUFFIX=("MelanoMix-Grid")
# DST_PROJECT="MelaNet-MelanoMix-Sweeps"


# SRC_PROJECTS=()
# for prj_suffix in "${SRC_PROJECTS_SUFFIX[@]}"; do
#     for prj_base in "${SRC_PROJECTS_BASE[@]}"; do
#         SRC_PROJECTS+=("${prj_base}-${prj_suffix}")
#     done
# done
# echo "${SRC_PROJECTS[@]}"

# ======================================================================================================================================================
# Ensembles
# SRC_PROJECTS=("Sweep-Grid-Attn-Ensemble-ViT-ResNet" "Sweep-Grid-Concat-Ensemble-ViT-ResNet" "Sweep-Grid-Attn-Ensemble-SwinV2-CAFormer" "Sweep-Grid-Concat-Ensemble-SwinV2-CAFormer")
# DST_PROJECT="MelaNet-Ensemble-Sweeps"

# ======================================================================================================================================================


for src_project in "${SRC_PROJECTS[@]}"; do
    echo "=============================================================================="
    echo "Copying runs from ${SRC_ENTITY}/${src_project} to ${DST_ENTITY}/${DST_PROJECT}"

    python wandb_copy_runs.py \
        --src-entity "${SRC_ENTITY}" \
        --src-project "${src_project}" \
        --dst-entity "${DST_ENTITY}" \
        --dst-project "${DST_PROJECT}"
done