# =========================================================================================================
# ViT + ResNet

MODEL_NAME_1="/home/sulcm/models/melanet/vit_base/melanet_vit_base_fine_grid-dxjjru4m"
MODEL_NAME_2="/home/sulcm/models/melanet/resnet50/melanet_resnet50_fine_grid-n5511yzn"
BATCH_SIZE=128

FUSION_NETWORK_BASE_PATH="/home/sulcm/models/melanet/adapters/fusion/vit_resnet50"
FUSION_NETWORK="ViT_base_ResNet50_fusion_attn-gwbeuyph.pt"
# ViT_base_ResNet50_fusion_attn-6ppg170u.pt
# ViT_base_ResNet50_fusion_attn-8ovv3wml.pt
# ViT_base_ResNet50_fusion_attn-mttp209h.pt
# ViT_base_ResNet50_fusion_attn-jv8xnnx8.pt
# ViT_base_ResNet50_fusion_attn-gwbeuyph.pt
# ViT_base_ResNet50_fusion_concat-plnceb75.pt
# ViT_base_ResNet50_fusion_concat-xahqtyx8.pt

# =========================================================================================================
# Swin V2 + CAFormer S18

# MODEL_NAME_1="/home/sulcm/models/melanet/swinv2_base/melanet_swinv2_base_grid-g3ki3vjd"
# MODEL_NAME_2="/home/sulcm/models/melanet/caformer_s18/melanet_caformer_s18_fine_grid-76v09gjq"
# BATCH_SIZE=16

# FUSION_NETWORK_BASE_PATH="/home/sulcm/models/melanet/adapters/fusion/swinv2_caformer"
# FUSION_NETWORK="swinv2_base_caformer_s18_fusion_concat-4119twa7.pt"
# # swinv2_base_caformer_s18_fusion_attn-2qbbgvwf.pt
# # swinv2_base_caformer_s18_fusion_attn-5ky6c913.pt
# # swinv2_base_caformer_s18_fusion_attn-8olop91h.pt
# # swinv2_base_caformer_s18_fusion_concat-4119twa7.pt
# # swinv2_base_caformer_s18_fusion_concat-tpbzauhm.pt

# =========================================================================================================

EVAL_TYPE="eval"
# EVAL_TYPE="test"

fusion_name="${FUSION_NETWORK_BASE_PATH##*/}"
fusion_model_name="${FUSION_NETWORK%.*}"
OUTPUT_DIR="./outputs/${fusion_name}_ensemble"
mkdir -p "$OUTPUT_DIR"

if [ "$EVAL_TYPE" = "eval" ]; then
    PREDICTION_STRATEGY="greedy"
    DATASET_PATHS=("/home/sulcm/datasets/milk10k/SpilledMILK10k" "/home/sulcm/datasets/ham10000/HAM10000")
    ID_COLUMN_NAME="isic_id" # isic_id, lesion_id

    for dataset_path in "${DATASET_PATHS[@]}"; do
        dataset_name="${dataset_path##*/}"

        fusion_path="fusion---${FUSION_NETWORK_BASE_PATH}/${FUSION_NETWORK}"
        output_path="${OUTPUT_DIR}/results_${dataset_name}_${fusion_model_name}_${PREDICTION_STRATEGY}.json"
        echo "Running ${fusion_path} ensemble (output ${output_path})"

        python run_eval.py \
            --classification_task "multiclass" \
            --model_name_or_path "${MODEL_NAME_1}" \
            --zero_shot_model_name_or_path "${MODEL_NAME_2}" \
            --zero_shot_config "model_backend=hf" \
            --models_fusion_adapter_name "${fusion_path}" \
            --dataset_name "${dataset_path}" \
            --eval_split "validation" \
            --prediction_resolution_strategy "${PREDICTION_STRATEGY}" \
            --batch_size $BATCH_SIZE \
            --results_path "${output_path}" \
            --id_column_name "${ID_COLUMN_NAME}" \
            --label_column_name "label"
    done
elif [ "$EVAL_TYPE" = "test" ]; then
    PREDICTION_STRATEGY="mean"
    DATASET_PATH="/home/sulcm/datasets/milk10k/TestMILK10k"

    fusion_path="fusion---${FUSION_NETWORK_BASE_PATH}/${FUSION_NETWORK}"
    output_path="${OUTPUT_DIR}/intermediate.json"
    submission_path="${OUTPUT_DIR}/results_${fusion_model_name}_${PREDICTION_STRATEGY}.csv"
    echo "Running test ${fusion_path} ensemble (submission ${submission_path})"

    python run_eval.py \
        --classification_task "multiclass" \
        --model_name_or_path "${MODEL_NAME_1}" \
        --zero_shot_model_name_or_path "${MODEL_NAME_2}" \
        --zero_shot_config "model_backend=hf" \
        --models_fusion_adapter_name "${fusion_path}" \
        --dataset_name "${DATASET_PATH}" \
        --eval_split "test" \
        --prediction_resolution_strategy "${PREDICTION_STRATEGY}" \
        --batch_size $BATCH_SIZE \
        --results_path "${output_path}" \
        --isic_submission_path "${submission_path}" \
        --id_column_name "lesion_id"
else
    echo "Unknown EVAL_TYPE ${EVAL_TYPE}, exiting ..."
    exit 1
fi