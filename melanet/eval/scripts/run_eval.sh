python run_eval.py \
    --classification_task "multiclass" \
    --model_name_or_path "/home/sulcm/models/melanet/swinv2_base/melanet_swinv2_base_melanomix_grid-6vwt2mvz" \
    --prediction_resolution_strategy "mean" \
    --dataset_name "/home/sulcm/datasets/milk10k/TestMILK10k" \
    --eval_split "test" \
    --results_path "./outputs/intermediate.json" \
    --batch_size 16 \
    --id_column_name "lesion_id" \
    --isic_submission_path "./outputs/swinv2_base_6vwt2mvz_melanomix_mean.csv" \
    # --label_column_name "label" \