OUTPUT_DIR="./outputs/ham10000_cls"
mkdir -p "$OUTPUT_DIR"

RESNET=("resnet50" "/home/sulcm/models/melanet/resnet50/melanet_resnet50_fine_grid-n5511yzn" 128)
CONVNEXT=("convnext_v2" "/home/sulcm/models/melanet/convnext_v2/melanet_convnext_v2_grid-6u43iqe4" 64)
EFFNET=("efficientnet_v2" "/home/sulcm/models/melanet/efficientnet_v2/melanet_efficientnet_v2_fine_grid-7idjqrjr" 128)
CAFORMER=("caformer_s18" "/home/sulcm/models/melanet/caformer_s18/melanet_caformer_s18_fine_grid-76v09gjq" 128)
VIT=("vit_base" "/home/sulcm/models/melanet/vit_base/melanet_vit_base_fine_grid-dxjjru4m" 128)
SWIN=("swinv2_base" "/home/sulcm/models/melanet/swinv2_base/melanet_swinv2_base_grid-g3ki3vjd" 16)
HVIT=("hybrid_vit" "/home/sulcm/models/melanet/google_hybrid_vit/melanet_hvit_grid-6x3p1an6" 128)

EVAL_MODELS=(RESNET CONVNEXT EFFNET CAFORMER VIT SWIN HVIT)

for model_eval_row in "${EVAL_MODELS[@]}"; do
    declare -n model_eval=$model_eval_row

    model_name=${model_eval[0]}
    model_path=${model_eval[1]}
    batch_size=${model_eval[2]}

    OUTPUT_PATH="${OUTPUT_DIR}/${model_name}.json"

    echo "Running $model_path at $batch_size"
    echo "Results will be saved to $OUTPUT_PATH"

    python run_eval.py \
        --classification_task "multiclass" \
        --model_name_or_path "${model_path}" \
        --prediction_resolution_strategy "mean" \
        --dataset_name "/home/sulcm/datasets/ham10000/HAM10000" \
        --eval_split "validation" \
        --results_path "${OUTPUT_PATH}" \
        --batch_size $batch_size \
        --id_column_name "lesion_id" \
        --label_column_name "label"
done