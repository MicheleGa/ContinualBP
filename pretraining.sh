# Pretraining

# Meta-learning Setup
# MAML BIOT
experiment_name="biot_maml"
mkdir "logs/$experiment_name"
cd ./models
python BIOT.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 10 \
    --pretrained_path ../checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.BIOT \
    --dataset_name mimic_iii_biot \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --sig2sig False \
    --fs 125 \
    --input_seq_len_s 10 \
    --pretrained_path ./checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
    --ecg True \
    --batch_size 128 \
    --mix_pretraining_subject_samples False \
    --lambda_supervised 1.0 \
    --criterion "SmoothL1Loss" \
    --freeze_backbone_first True \
    --pre_train_lr 0.001 \
    --backbone_lr_multiplier 1.0 \
    --weight_decay 0.0001 \
    --grad_clip 10.0 \
    --ft_stage1_epochs 10 \
    --ft_stage2_epochs 25 \
    --warmup_epochs_stage1 3 \
    --warmup_epochs_stage2 5 \
    --meta_learning True \
    --meta_algorithm 'maml' \
    --max_meta_epochs 800 \
    --k_support 8 \
    --k_query 8 \
    --meta_batch_size 8 \
    --use_pure_functional True \
    --second_order_maml True \
    --meta_lr_schedule 'cosine_wr' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_T0 100 \
    --meta_lr_scheduler_T_mult 1.5 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --meta_lr_scheduler_gamma 0.7 \
    --meta_lr_scheduler_min_gamma 1.0 \
    --meta_lr_scheduler_max_cycles 3 \
    --meta_lr_scheduler_tail 'cosine' \
    --meta_lr_scheduler_eta_floor 0.00001 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_head_lr_mult 1.0 \
    --inner_lr_schedule 'cosine' \
    --inner_lr 0.01 \
    --inner_lr_min 0.001 \
    --inner_steps_schedule 'cosine' \
    --inner_steps 4 \
    --inner_steps_max 8 \
    --eval_lr 0.005 \
    --eval_steps 5 \
    > "./logs/$experiment_name/${experiment_name}_training.log"

