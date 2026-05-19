# =========================================================================================================
# DermLIP ViT-B/16

ZERO_SHOT_MODEL_NAME="hf-hub:redlessone/DermLIP_ViT-B-16"
BACKEND_TYPE="open_clip"
BATCH_SIZE=128

LIN_PROBE_BASE_PATH="/home/sulcm/models/melanet/adapters/linear/dermlip_vit"
LIN_PROBE="DermLIP_aug_lin_probe_01.pt"
# DermLIP_aug_lin_probe_01.pt # selected

# =========================================================================================================
# DINOv3 ViT-H+/16

# ZERO_SHOT_MODEL_NAME="timm/vit_huge_plus_patch16_dinov3.lvd1689m"
# BACKEND_TYPE="timm"
# BATCH_SIZE=32

# LIN_PROBE_BASE_PATH="/home/sulcm/models/melanet/adapters/linear/dinov3_vit"
# LIN_PROBE="DINOv3_aug_lin_probe_02.pt"
# # DINOv3_aug_lin_probe_01.pt
# # DINOv3_aug_lin_probe_02.pt # selected
# # test_DINOv3_aug_lin_probe_01.pt - lr: 1.25e-4 e:40
# # test_DINOv3_aug_lin_probe_02.pt - lr: 2.5e-4 e:30
# # test_DINOv3_aug_lin_probe_03.pt - lr: 2e-4 e:30
# # test_DINOv3_aug_lin_probe_04.pt - lr: 2e-4 e:40
# # test_DINOv3_aug_lin_probe_05.pt - lr: 1.5e-4 e:40
# # test_DINOv3_aug_lin_probe_06.pt - lr: 1.75e-4 e:40

# =========================================================================================================

EVAL_TYPE="eval"
# EVAL_TYPE="test"

model_probe_name="${LIN_PROBE_BASE_PATH##*/}"
lin_probe_model_name="${LIN_PROBE%.*}"
OUTPUT_DIR="./outputs/${model_probe_name}_lin_probe"
mkdir -p "$OUTPUT_DIR"

if [ "$EVAL_TYPE" = "eval" ]; then
    PREDICTION_STRATEGY="greedy"
    DATASET_PATHS=("/home/sulcm/datasets/milk10k/SpilledMILK10k" "/home/sulcm/datasets/ham10000/HAM10000")
    ID_COLUMN_NAME="isic_id" # isic_id, lesion_id

    for dataset_path in "${DATASET_PATHS[@]}"; do
        dataset_name="${dataset_path##*/}"

        lin_probe_path="linear---${LIN_PROBE_BASE_PATH}/${LIN_PROBE}"
        output_path="${OUTPUT_DIR}/results_${dataset_name}_${lin_probe_model_name}_${PREDICTION_STRATEGY}.json"
        echo "Running ${lin_probe_path} linear probe (output ${output_path})"

        python run_eval.py \
            --classification_task "multiclass" \
            --zero_shot_model_name_or_path "${ZERO_SHOT_MODEL_NAME}" \
            --zero_shot_config "model_backend=${BACKEND_TYPE}" \
            --zero_shot_model_adapter_name "${lin_probe_path}" \
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

    lin_probe_path="linear---${LIN_PROBE_BASE_PATH}/${LIN_PROBE}"
    output_path="${OUTPUT_DIR}/intermediate.json"
    submission_path="${OUTPUT_DIR}/results_${lin_probe_model_name}_${PREDICTION_STRATEGY}.csv"
    echo "Running test ${lin_probe_path} linear probe (submission ${submission_path})"

    python run_eval.py \
        --classification_task "multiclass" \
        --zero_shot_model_name_or_path "${ZERO_SHOT_MODEL_NAME}" \
        --zero_shot_config "model_backend=${BACKEND_TYPE}" \
        --zero_shot_model_adapter_name "${lin_probe_path}" \
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