python run_eval.py \
    --classification_task "multiclass" \
    --model_name_or_path "/home/sulcm/models/melanet/vit_base/melanet_vit_base_grid-xs5p4lv0" \
    --dataset_name "/home/sulcm/datasets/milk10k/TestMILK10k" \
    --eval_split "test" \
    --results_path "./outputs/intermediate.json" \
    --batch_size 128 \
    --id_column_name "lesion_id" \
    --isic_submission_path "./outputs/vit_base_xs5p4lv0_grid.csv" \
    # --label_column_name "label" \