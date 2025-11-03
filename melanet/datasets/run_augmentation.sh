python run_augmentation.py \
    --dataset_name /home/sulcm/datasets/milk10k/SpilledMILK10k \
    --augmented_dataset_path /home/sulcm/datasets/milk10k/AugmentedMILK10k \
    --dataset_split train \
    --batch_size 512 \
    --num_proc 4 \
    --image_column_name image \
    --label_column_name label \
    --seed 42