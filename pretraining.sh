# Pretraining

## Supervised Efficient UNet
experiment_name="efficient_unet_test"
mkdir "logs/$experiment_name"
cd ./models
python EUNet.py \
    --sig2sig True \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --batch_size 128 \
    --channels "16,32,64,128" \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.EUNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --sig2sig True \
    --fs 125 \
    --input_seq_len_s 5 \
    --ecg True \
    --batch_size 128 \
    --channels "16,32,64,128" \
    --mix_pretraining_subject_samples True \
    --max_training_epochs 1 \
    --lr_scheduler_enable True \
    --optimizer_type "Adam" \
    --lr_scheduler_type "ExponentialLR" \
    --criterion "SmoothL1Loss" \
    --eval_every_n_epochs 1 \
    --lambda_supervised 1 \
    > "./logs/$experiment_name/${experiment_name}_training.log"


## Self-Supervised Efficient UNet 
experiment_name="ssl_eunet_test"
mkdir "logs/$experiment_name"
cd ./models
python SSLEUNet.py \
    --sig2sig True \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --batch_size 128 \
    --channels "16,32,64,128" \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.SSLEUNet \
    --dataset_name mimic_iii_xl \
    --lp_dataset_name mimic_iii_xl \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --sig2sig True \
    --ecg True \
    --ssl True \
    --fs 125 \
    --input_seq_len_s 5 \
    --channels "16,32,64,128" \
    --batch_size 64 \
    --mix_pretraining_subject_samples False \
    --max_training_epochs 1 \
    --max_lp_training_epochs 1 \
    --lr_scheduler_enable True \
    --optimizer_type "Adam" \
    --lr_scheduler_type "ExponentialLR" \
    --criterion "SmoothL1Loss" \
    --eval_every_n_epochs 1 \
    --apply_masking True \
    --lambda_msr 1 \
    --masking_ratio 0.08 \
    > "./logs/$experiment_name/${experiment_name}_training.log"