## Personalization

# GRU
experiment_name="personalization_vanilla_gru"
mkdir "logs/$experiment_name"
cd ./models
python GRU.py \
    --fs 125 \
    --input_seq_len_s 5 \
    --ecg True \
    --hidden_dim 128 \
    --num_layers 2 \
    --bidirectional True \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.GRU \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --pretrained_model_checkpoint ./checkpoints/vanilla_gru/vanilla_gru-GRU-2025_05_28-17_39_44/vanilla_gru/ckpt/GRU \
    --loader_worker 4 \
    --sig2sig True \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --hidden_dim 128 \
    --num_layers 2 \
    --bidirectional True \
    --mix_pretraining_subject_samples True \
    --max_training_epochs 1000 \
    --optimizer_type "Adam" \
    --lr_scheduler_enable False \
    --criterion "SmoothL1Loss" \
    --eval_every_n_epochs 1 \
    --batch_size 32 \
    --es_enable True \
    --es_patience 20 \
    --es_min_delta 0.005 \
    --personalization_sample_number 100 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    > "./logs/$experiment_name/${experiment_name}_training.log"

# UNet
experiment_name="personalization_unet"
mkdir "logs/$experiment_name"
cd ./models
python UNet.py \
    --fs 125 \
    --input_seq_len_s 5 \
    --ecg True \
    --num_heads_attention 1 \
    --dim_feedforward_attention 128 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.UNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --pretrained_model_checkpoint ./checkpoints/unet/unet-UNet-2025_05_28-23_19_19/unet/ckpt/UNet \
    --loader_worker 4 \
    --sig2sig True \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --ecg True \
    --num_heads_attention 1 \
    --dim_feedforward_attention 128 \
    --mix_pretraining_subject_samples True \
    --max_training_epochs 1000 \
    --optimizer_type "Adam" \
    --lr_scheduler_enable False \
    --criterion "SmoothL1Loss" \
    --eval_every_n_epochs 1 \
    --batch_size 32 \
    --es_enable True \
    --es_patience 20 \
    --es_min_delta 0.005 \
    --personalization_sample_number 100 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    > "./logs/$experiment_name/${experiment_name}_training.log"