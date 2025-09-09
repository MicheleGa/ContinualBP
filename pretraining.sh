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
    --freeze_backbone_first True \
    --pre_train_lr 0.001 \
    --backbone_lr_multiplier 1.0 \
    --weight_decay 0.0001 \
    --grad_clip 10.0 \
    --ft_stage1_epochs 10 \
    --ft_stage2_epochs 25 \
    --warmup_epochs_stage1 3 \
    --warmup_epochs_stage2 5 \
    --criterion "SmoothL1Loss" \
    --lambda_supervised 0.5 \
    --lambda_contrastive 1.0 \
    --temperature 0.07 \
    --meta_learning True \
    --meta_algorithm 'maml' \
    --max_meta_epochs 150 \
    --k_support 5 \
    --k_query 10 \
    --meta_batch_size 4 \
    --use_pure_functional True \
    --second_order_maml True \
    --meta_lr_schedule 'multistep' \
    --meta_lr 0.001 \
    --meta_lr_gamma 0.7 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_head_lr_mult 1.0 \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.005 \
    --inner_steps_schedule 'constant' \
    --inner_steps 8 \
    --eval_lr 0.005 \
    --eval_steps 8 \
    > "./logs/$experiment_name/${experiment_name}_training.log"