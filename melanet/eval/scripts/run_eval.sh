python run_eval.py \
    --classification_task "multiclass" \
    --model_name_or_path "/home/sulcm/models/melanet/resnet50/melanet_resnet50_fine_grid-n5511yzn" \
    --prediction_resolution_strategy "mean" \
    --dataset_name "/home/sulcm/datasets/milk10k/TestMILK10k" \
    --eval_split "test" \
    --results_path "./outputs/intermediate.json" \
    --batch_size 128 \
    --id_column_name "lesion_id" \
    --isic_submission_path "./outputs/resnet50_n5511yzn_mean.csv" \
    # --label_column_name "label" \