# Pretraining

# Meta-learning Setup
## Reptile BIOT
experiment_name="biot_maml_test"
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
    --base_lr 0.001 \
    --backbone_lr_multiplier 1.0 \
    --weight_decay 0.0001 \
    --grad_clip 10.0 \
    --ft_stage1_epochs 10 \
    --ft_stage2_epochs 25 \
    --warmup_epochs_stage1 3 \
    --warmup_epochs_stage2 5 \
    --freeze_backbone_first True \
    --meta_learning True \
    --first_order_reptile True \
    --inner_adapt 'head' \
    --inner_opt 'sgd' \
    --sgd_momentum 0.0 \
    --inner_head_lr_mult 1.0 \
    --inner_steps 5 \
    --meta_lr 0.001 \
    --lr_inner 0.001 \
    --max_training_epochs 500 \
    --meta_lr_schedule 'cosine' \
    --inner_lr_schedule 'constant' \
    --inner_steps_schedule 'constant' \
    --criterion "SmoothL1Loss" \
    --k_support 5 \
    --k_query 5 \
    --meta_batch_size 8 \
    --use_pure_functional True \
    --second_order_maml  True \
    > "./logs/$experiment_name/${experiment_name}_training.log"


## Supervised Pretrained BIOT
#experiment_name="pretrained_biot"
#mkdir "logs/$experiment_name"
#cd ./models
#python BIOT.py \
#    --ecg True \
#    --sig2sig True \
#    --fs 200 \
#    --input_seq_len_s 10 \
#    --pretrained_path ../checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python pretraining.py \
#    --model models.BIOT \
#    --dataset_name mimic_iii_biot \
#    --expname "$experiment_name" \
#    --loader_worker 4 \
#    --sig2sig True \
#    --fs 200 \
#    --input_seq_len_s 10 \
#    --ecg True \
#    --batch_size 128 \
#    --mix_pretraining_subject_samples True \
#    --pretrained_path ./checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
#    --criterion "SmoothL1Loss" \
#    --eval_every_n_epochs 1 \
#    --lambda_supervised 1 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"

### Supervised Efficient UNet
#experiment_name="efficient_unet_test"
#mkdir "logs/$experiment_name"
#cd ./models
#python EUNet.py \
#    --sig2sig True \
#    --ecg True \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --batch_size 128 \
#    --channels "16,32,64,128" \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python pretraining.py \
#    --model models.EUNet \
#    --dataset_name mimic_iii \
#    --expname "$experiment_name" \
#    --loader_worker 4 \
#    --sig2sig True \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --ecg True \
#    --batch_size 128 \
#    --channels "16,32,64,128" \
#    --mix_pretraining_subject_samples True \
#    --max_training_epochs 1 \
#    --lr_scheduler_enable True \
#    --optimizer_type "Adam" \
#    --lr_scheduler_type "ExponentialLR" \
#    --criterion "SmoothL1Loss" \
#    --eval_every_n_epochs 1 \
#    --lambda_supervised 1 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"
#
#
### Self-Supervised Efficient UNet 
#experiment_name="ssl_eunet_test"
#mkdir "logs/$experiment_name"
#cd ./models
#python SSLEUNet.py \
#    --sig2sig True \
#    --ecg True \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --batch_size 128 \
#    --channels "16,32,64,128" \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python pretraining.py \
#    --model models.SSLEUNet \
#    --dataset_name mimic_iii_xl \
#    --lp_dataset_name mimic_iii_xl \
#    --expname "$experiment_name" \
#    --loader_worker 4 \
#    --sig2sig True \
#    --ecg True \
#    --ssl True \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --channels "16,32,64,128" \
#    --batch_size 64 \
#    --mix_pretraining_subject_samples False \
#    --max_training_epochs 1 \
#    --max_lp_training_epochs 1 \
#    --lr_scheduler_enable True \
#    --optimizer_type "Adam" \
#    --lr_scheduler_type "ExponentialLR" \
#    --criterion "SmoothL1Loss" \
#    --eval_every_n_epochs 1 \
#    --apply_masking True \
#    --lambda_msr 1 \
#    --masking_ratio 0.08 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"