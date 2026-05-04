ROTATE_AUGMENTATIONS=("15" "90" "270" "345")

DINOV3=("dinov3_vit" "timm" "timm/vit_huge_plus_patch16_dinov3.lvd1689m" 128)
DERMLIP=("dermlip_vit" "open_clip" "hf-hub:redlessone/DermLIP_ViT-B-16" 256)
EVAL_MODELS=(DINOV3 DERMLIP)

SAVE_DIR="/storage/plzen4-ntis/home/sulcm01/datasets/melanet"

DATASET_PATHS=("metacentrum_scratch/SpilledMILK10k" "metacentrum_scratch/HAM10000")

for model_eval_row in "${EVAL_MODELS[@]}"; do
    declare -n model_eval=$model_eval_row
    model_type=${model_eval[0]}
    model_backend=${model_eval[1]}
    model_path=${model_eval[2]}
    batch_size=${model_eval[3]}

    for dataset_path in "${DATASET_PATHS[@]}"; do
        dataset_name="${dataset_path##*/}"
        for rot_deg in "${ROTATE_AUGMENTATIONS[@]}"; do
            rot_augment="rotate_${rot_deg}"

            python run_eval.py \
                --classification_task "multiclass" \
                --run_inference_only True \
                --eval_as_feature_extraction True \
                --zero_shot_model_name_or_path "${model_path}" \
                --zero_shot_config "model_backend=${model_backend}" \
                --dataset_name "${dataset_path}" \
                --eval_split "validation" \
                --index_name "${dataset_path}" \
                --index_split "train" \
                --id_column_name "isic_id" \
                --label_column_name "label" \
                --batch_size $batch_size \
                --cache_model_inference "${SAVE_DIR}/${dataset_name}_${model_type}_${rot_augment}_aug_features.pkl" \
                --index_augmentations "${rot_augment}" \
                --test_time_augmentations "${rot_augment}" \
                --augmentations_keep_original False
        done
    done
done