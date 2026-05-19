MODEL_NAME="dermlip_vit"
MODEL_PATH="hf-hub:redlessone/DermLIP_ViT-B-16"

PREDICTION_STRATEGY="rrf"
TEST_TIME_AUGMENTATIONS=true
RETRIEVE_UNIQUE_ONLY="False"
RETRIEVE_TOP_K="10"
BATCH_SIZE=128

OUTPUT_DIR="./outputs/${MODEL_NAME}"
mkdir -p "$OUTPUT_DIR"

AUGMENTATIONS=("none" "center_crop" "horizontal_flip" "vertical_flip" "rotate_15" "rotate_90" "rotate_270" "rotate_345" "all")
for augment in "${AUGMENTATIONS[@]}"; do
    OUTPUT_PATH="${OUTPUT_DIR}/result_top_k_unique_${RETRIEVE_UNIQUE_ONLY}_${PREDICTION_STRATEGY}_${augment}.json"
    tta="none"
    if [ "$TEST_TIME_AUGMENTATIONS" = true ] ; then
        tta="${augment}"
        OUTPUT_PATH="${OUTPUT_DIR}/result_top_k_unique_${RETRIEVE_UNIQUE_ONLY}_${PREDICTION_STRATEGY}_tta_${augment}.json"
    fi

    echo "Processing $augment (with TTA $tta, top-k=$RETRIEVE_TOP_K, unique only=$RETRIEVE_UNIQUE_ONLY) -> $OUTPUT_PATH"

    python run_eval.py \
        --classification_task "multiclass" \
        --eval_as_feature_extraction True \
        --zero_shot_model_name_or_path "${MODEL_PATH}" \
        --zero_shot_config "model_backend=open_clip" \
        --dataset_name "/home/sulcm/datasets/milk10k/SpilledMILK10k" \
        --eval_split "validation" \
        --index_name "/home/sulcm/datasets/milk10k/SpilledMILK10k" \
        --index_split "train" \
        --id_column_name "lesion_id" \
        --label_column_name "label" \
        --prediction_resolution_strategy "${PREDICTION_STRATEGY}" \
        --results_path "${OUTPUT_PATH}" \
        --batch_size $BATCH_SIZE \
        --feature_extractor_config "normalize_output=True" \
        --vector_store_config "metric=ip,top_k=${RETRIEVE_TOP_K},unique_only=${RETRIEVE_UNIQUE_ONLY}" \
        --index_augmentations "${augment}" \
        --test_time_augmentations "${tta}"
done