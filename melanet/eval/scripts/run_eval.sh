python run_eval.py \
    --classification_task "multiclass" \
    --model_name_or_path "/storage/plzen4-ntis/home/sulcm01/outputs/swinv2_base/melanet_swinv2_base_ce_reg_aug-38iimmkp" \
    --dataset_name "metacentrum_scratch/TestMILK10k" \
    --eval_split "test" \
    --results_path "./outputs/intermediate.json" \
    --batch_size 128 \
    --id_column_name "lesion_id" \
    --isic_submission_path "./outputs/swinv2_base_38iimmkp_baseline.csv" \
    # --label_column_name "label" \