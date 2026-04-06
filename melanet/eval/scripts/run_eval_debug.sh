python run_eval.py \
    --classification_task "multiclass" \
    --model_name_or_path "/home/sulcm/models/melanet/resnet50/melanet_resnet50_grid-cdvq05pb" \
    --dataset_name "/home/sulcm/datasets/milk10k/SpilledMILK10k" \
    --eval_split "validation" \
    # --dataset_name "/home/sulcm/datasets/milk10k/TestMILK10k" \
    # --eval_split "test" \
    --prediction_resolution_strategy "mean" \
    --results_path "./outputs/debug.json" \
    --batch_size 128 \
    --id_column_name "lesion_id" \
    --isic_submission_path "./outputs/debug.csv" \
    --label_column_name "label" \