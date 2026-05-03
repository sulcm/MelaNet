# Models
# =========================================================================================================

# MODEL_NAME="dermlip_vit"
# MODEL_PATH="hf-hub:redlessone/DermLIP_ViT-B-16"
# BACKEND_TYPE="open_clip" # timm, open_clip, hf

# =========================================================================================================

MODEL_NAME="dinov3_vit"
MODEL_PATH="timm/vit_huge_plus_patch16_dinov3.lvd1689m"
BACKEND_TYPE="timm" # timm, open_clip, hf

# =========================================================================================================

OUTPUT_DIR="./outputs/${MODEL_NAME}_pred_strategies"
mkdir -p "$OUTPUT_DIR"

DATASET_PATHS=("/home/sulcm/datasets/milk10k/SpilledMILK10k" "/home/sulcm/datasets/ham10000/HAM10000")
APPLIED_AUGMENT_COMBINATIONS=("index" "index_query" "query")

PREDICTION_STRATEGIES=("greedy" "rrf" "rrf_top_k_unique" "rrf_top_k")
BASE_VS_CONFIG="metric=ip,l2_normalize=True"

echo "Starting ablation on prediction strategies using ${MODEL_NAME} model"

for dataset_path in "${DATASET_PATHS[@]}"; do
    dataset_name="${dataset_path##*/}"

    for augmented in "${APPLIED_AUGMENT_COMBINATIONS[@]}"; do
        if [ "$augmented" = "index" ]; then
            load_cached_model_inference="${dataset_path}_${MODEL_NAME}_aug_features.pkl"
            load_cached_model_inference_to_eval="${dataset_path}_${MODEL_NAME}_features.pkl"
        elif [ "$augmented" = "index_query" ]; then
            load_cached_model_inference="${dataset_path}_${MODEL_NAME}_aug_features.pkl"
            load_cached_model_inference_to_eval="${dataset_path}_${MODEL_NAME}_aug_features.pkl"
        elif [ "$augmented" = "query" ]; then
            load_cached_model_inference="${dataset_path}_${MODEL_NAME}_features.pkl"
            load_cached_model_inference_to_eval="${dataset_path}_${MODEL_NAME}_aug_features.pkl"
        else
            echo "Unknown augment combination ${augmented}, exiting ..."
            exit 1
        fi

        for pred_strategy_type in "${PREDICTION_STRATEGIES[@]}"; do
            if [ "$pred_strategy_type" = "greedy" ]; then
                pred_strategy="greedy"
                vector_store_config="${BASE_VS_CONFIG}"
            elif [ "$pred_strategy_type" = "rrf" ]; then
                pred_strategy="rrf"
                vector_store_config="${BASE_VS_CONFIG}"
            elif [ "$pred_strategy_type" = "rrf_top_k_unique" ]; then
                pred_strategy="rrf"
                vector_store_config="${BASE_VS_CONFIG},top_k=10,unique_only=True"
            elif [ "$pred_strategy_type" = "rrf_top_k" ]; then
                pred_strategy="rrf"
                vector_store_config="${BASE_VS_CONFIG},top_k=10,unique_only=False"
            else
                echo "Unknown prediction strategy ${pred_strategy_type}, exiting ..."
                exit 1
            fi

            OUTPUT_PATH="${OUTPUT_DIR}/results_${dataset_name}_${augmented}_${pred_strategy_type}.json"
            echo ""
            echo "Running ${pred_strategy_type} strategy with augmented ${augmented} on dataset ${dataset_name}"
            echo "Output will be saved to ${OUTPUT_PATH}"

            python run_eval.py \
                --classification_task "multiclass" \
                --eval_as_feature_extraction True \
                --zero_shot_model_name_or_path "${MODEL_PATH}" \
                --zero_shot_config "model_backend=${BACKEND_TYPE}" \
                --dataset_name "${dataset_path}" \
                --eval_split "validation" \
                --index_name "${dataset_path}" \
                --index_split "train" \
                --load_cached_model_inference "${load_cached_model_inference}" \
                --load_cached_model_inference_to_eval "${load_cached_model_inference_to_eval}" \
                --id_column_name "lesion_id" \
                --label_column_name "label" \
                --cached_id_column_name "isic_id" \
                --prediction_resolution_strategy "${pred_strategy}" \
                --vector_store_config "${vector_store_config}" \
                --results_path "${OUTPUT_PATH}"
        done
    done
done