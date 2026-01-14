# Pretraining

# BIOT w/ Supervised + MAML Pre-training

# PPG
experiment_name="biot_ppg_percentile_256_pretrained"
mkdir "logs/$experiment_name"
cd ./models
python BIOT.py \
    --batch_size 16 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 256 \
    --pretrained_encoder_ckpt_path ../checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.BIOT \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 256 \
    --pretrained_encoder_ckpt_path ./checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
    --batch_size 128 \
    --stage1_pre_train_lr 0.001 \
    --stage1_pre_train_scheduler_eta_min 0.00001 \
    --weight_decay 0.0001 \
    --stage1_epochs 10 \
    --criterion "SmoothL1Loss" \
    --max_meta_epochs 200 \
    --k_support 16 \
    --k_query 16 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.01 \
    --inner_steps_schedule 'constant' \
    --inner_steps 8 \
    --eval_lr 0.01 \
    --eval_steps 8 \
    > "./logs/$experiment_name/${experiment_name}_training.log"

experiment_name="biot_ppg_z_score_256_pretrained"
mkdir "logs/$experiment_name"
cd ./models
python BIOT.py \
    --batch_size 16 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 256 \
    --pretrained_encoder_ckpt_path ../checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.BIOT \
    --dataset_name pulse_db_mimic_iii_z_score \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 256 \
    --pretrained_encoder_ckpt_path ./checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
    --batch_size 128 \
    --stage1_pre_train_lr 0.001 \
    --stage1_pre_train_scheduler_eta_min 0.00001 \
    --weight_decay 0.0001 \
    --stage1_epochs 10 \
    --criterion "SmoothL1Loss" \
    --max_meta_epochs 200 \
    --k_support 16 \
    --k_query 16 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.01 \
    --inner_steps_schedule 'constant' \
    --inner_steps 8 \
    --eval_lr 0.01 \
    --eval_steps 8 \
    > "./logs/$experiment_name/${experiment_name}_training.log"
