python run_eval.py \
    --classification_task "multiclass" \
    --model_name_or_path "/home/sulcm/models/resnet50/melanet_resnet50_seesaw-m4fnomwn" \
    --dataset_name "/home/sulcm/datasets/milk10k/TestMILK10k" \
    --dataset_split "test" \
    --results_path "./outputs/test.json" \
    --batch_size 128 \
    # --label_column_name "label" \