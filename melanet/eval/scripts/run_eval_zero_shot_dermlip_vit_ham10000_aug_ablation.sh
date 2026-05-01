MODEL_NAME="dermlip_vit_ham10000"
MODEL_PATH="hf-hub:redlessone/DermLIP_ViT-B-16"

# DATASET_PATH="/home/sulcm/datasets/ham10000/HAM10000"
DATASET_PATH="metacentrum_scratch/HAM10000"

PREDICTION_STRATEGY="greedy"
BATCH_SIZE=256

OUTPUT_DIR="./outputs/${MODEL_NAME}"
mkdir -p "$OUTPUT_DIR"

AUGMENTATIONS=("center_crop" "horizontal_flip" "vertical_flip" "rotate_15" "rotate_90" "rotate_270" "rotate_345" "all")
DO_TTA=(false true)
for augment in "${AUGMENTATIONS[@]}"; do
    # With Index augment
    for do_tta_iter in "${DO_TTA[@]}"; do
        OUTPUT_PATH="${OUTPUT_DIR}/result_${PREDICTION_STRATEGY}_${augment}.json"
        tta="none"
        if [ "$do_tta_iter" = true ] ; then
            tta="${augment}"
            OUTPUT_PATH="${OUTPUT_DIR}/result_${PREDICTION_STRATEGY}_tta_${augment}.json"
        fi

        echo "Processing $augment (with TTA $tta) -> $OUTPUT_PATH"

        python run_eval.py \
            --classification_task "multiclass" \
            --eval_as_feature_extraction True \
            --zero_shot_model_name_or_path "${MODEL_PATH}" \
            --zero_shot_config "model_backend=open_clip" \
            --dataset_name "${DATASET_PATH}" \
            --eval_split "validation" \
            --index_name "${DATASET_PATH}" \
            --index_split "train" \
            --id_column_name "lesion_id" \
            --label_column_name "label" \
            --prediction_resolution_strategy "${PREDICTION_STRATEGY}" \
            --results_path "${OUTPUT_PATH}" \
            --batch_size $BATCH_SIZE \
            --feature_extractor_config "normalize_output=True" \
            --vector_store_config "metric=ip" \
            --index_augmentations "${augment}" \
            --test_time_augmentations "${tta}"
    done

    # With OUT Index augment but with TTA
    tta="${augment}"
    OUTPUT_PATH="${OUTPUT_DIR}/result_${PREDICTION_STRATEGY}_tta_${augment}_no_index.json"
    echo "Processing without index augment (with TTA $tta) -> $OUTPUT_PATH"

    python run_eval.py \
        --classification_task "multiclass" \
        --eval_as_feature_extraction True \
        --zero_shot_model_name_or_path "${MODEL_PATH}" \
        --zero_shot_config "model_backend=open_clip" \
        --dataset_name "${DATASET_PATH}" \
        --eval_split "validation" \
        --index_name "${DATASET_PATH}" \
        --index_split "train" \
        --id_column_name "lesion_id" \
        --label_column_name "label" \
        --prediction_resolution_strategy "${PREDICTION_STRATEGY}" \
        --results_path "${OUTPUT_PATH}" \
        --batch_size $BATCH_SIZE \
        --feature_extractor_config "normalize_output=True" \
        --vector_store_config "metric=ip" \
        --index_augmentations "none" \
        --test_time_augmentations "${tta}"
done

# Baseline without augments
OUTPUT_PATH="${OUTPUT_DIR}/result_${PREDICTION_STRATEGY}_none.json"
echo "Processing baseline (without augment) -> $OUTPUT_PATH"

python run_eval.py \
    --classification_task "multiclass" \
    --eval_as_feature_extraction True \
    --zero_shot_model_name_or_path "${MODEL_PATH}" \
    --zero_shot_config "model_backend=open_clip" \
    --dataset_name "${DATASET_PATH}" \
    --eval_split "validation" \
    --index_name "${DATASET_PATH}" \
    --index_split "train" \
    --id_column_name "lesion_id" \
    --label_column_name "label" \
    --prediction_resolution_strategy "${PREDICTION_STRATEGY}" \
    --results_path "${OUTPUT_PATH}" \
    --batch_size $BATCH_SIZE \
    --feature_extractor_config "normalize_output=True" \
    --vector_store_config "metric=ip" \
    --index_augmentations "none" \
    --test_time_augmentations "none"