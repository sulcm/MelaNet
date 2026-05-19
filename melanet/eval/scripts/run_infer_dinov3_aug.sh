AUGMENTATIONS="horizontal_flip,vertical_flip"

MODEL_TYPE="dinov3_vit"
MODEL_PATH="timm/vit_huge_plus_patch16_dinov3.lvd1689m"
SAVE_DIR="/storage/plzen4-ntis/home/sulcm01/datasets/melanet"
BATCH_SIZE=128

DATASET_PATHS=("metacentrum_scratch/SpilledMILK10k" "metacentrum_scratch/HAM10000")
for dataset_path in "${DATASET_PATHS[@]}"; do
    dataset_name="${dataset_path##*/}"
    python run_eval.py \
        --classification_task "multiclass" \
        --run_inference_only True \
        --eval_as_feature_extraction True \
        --zero_shot_model_name_or_path "${MODEL_PATH}" \
        --zero_shot_config "model_backend=timm" \
        --dataset_name "${dataset_path}" \
        --eval_split "validation" \
        --index_name "${dataset_path}" \
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
        --zero_shot_model_name_or_path "${MODEL_PATH}" \
        --zero_shot_config "model_backend=timm" \
        --dataset_name "${dataset_path}" \
        --eval_split "validation" \
        --index_name "${dataset_path}" \
        --index_split "train" \
        --id_column_name "isic_id" \
        --label_column_name "label" \
        --batch_size $BATCH_SIZE \
        --cache_model_inference "${SAVE_DIR}/${dataset_name}_${MODEL_TYPE}_features.pkl"
done