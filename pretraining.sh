# Pretraining

# Meta-learning Setup
# MAML BIOT
experiment_name="biot_maml_pretraining"
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
    --pre_train_lr 0.001 \
    --weight_decay 0.0001 \
    --ft_stage1_epochs 300 \
    --ft_stage2_epochs 150 \
    --criterion "SmoothL1Loss" \
    --temperature 0.07 \
    --meta_learning True \
    --meta_algorithm 'maml' \
    --max_meta_epochs 150 \
    --grad_clip 10.0 \
    --k_support 5 \
    --k_query 10 \
    --meta_batch_size 8 \
    --use_pure_functional True \
    --second_order_maml True \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --derivative_order_anneal_epoch 80 \
    --msl_anneal_epochs 80 \
    --msl_include_pre False \
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