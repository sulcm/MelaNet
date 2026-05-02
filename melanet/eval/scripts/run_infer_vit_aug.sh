AUGMENTATIONS="horizontal_flip,vertical_flip"

MODEL_TYPE="vit_base"
MODEL_PATH="metacentrum_scratch/melanet_vit_base_fine_grid-dxjjru4m"
SAVE_DIR="/storage/plzen4-ntis/home/sulcm01/datasets/melanet"
BATCH_SIZE=256

DATASET_PATHS=("metacentrum_scratch/SpilledMILK10k" "metacentrum_scratch/HAM10000")
for dataset_path in "${DATASET_PATHS[@]}"; do
    dataset_name="${dataset_path##*/}"
    python run_eval.py \
        --classification_task "multiclass" \
        --run_inference_only True \
        --eval_as_feature_extraction True \
        --model_name_or_path "${MODEL_PATH}" \
        --dataset_name "${DATASET_PATH}" \
        --eval_split "validation" \
        --index_name "${DATASET_PATH}" \
        --index_split "train" \
        --id_column_name "isic_id" \
        --label_column_name "label" \
        --batch_size $BATCH_SIZE \
        --cache_model_inference "${SAVE_DIR}/${dataset_name}_${MODEL_TYPE}_aug_features.pkl" \
        --index_augmentations "${AUGMENTATIONS}" \
        --test_time_augmentations "${AUGMENTATIONS}"
    #
    #
    python run_eval.py \
        --classification_task "multiclass" \
        --run_inference_only True \
        --eval_as_feature_extraction True \
        --model_name_or_path "${MODEL_PATH}" \
        --dataset_name "${DATASET_PATH}" \
        --eval_split "validation" \
        --index_name "${DATASET_PATH}" \
        --index_split "train" \
        --id_column_name "isic_id" \
        --label_column_name "label" \
        --batch_size $BATCH_SIZE \
        --cache_model_inference "${SAVE_DIR}/${dataset_name}_${MODEL_TYPE}_features.pkl"
done