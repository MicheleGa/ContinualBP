# Vanilla Resnet w/ PPG + ECG 
experiment_name="vanilla_resnet"
mkdir "logs/$experiment_name"
cd ./models
python ResGRUNet.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.ResGRUNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    --mix_pretraining_subject_samples True \
    --max_training_epochs 50 \
    --lr_scheduler_enable True \
    --optimizer_type "AdamW" \
    --lr_scheduler_type "CosineAnnealingWarmupScheduler" \
    --lr 0.01 \
    --lr_scheduler_min_lr 0.00001 \
    --lr_scheduler_warmup 10 \
    --eval_every_n_epochs 1 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    --aug False \
    > "./logs/$experiment_name/${experiment_name}_training.log"
