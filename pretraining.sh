# Pretraining

## Supervised Efficient UNet
#experiment_name="efficient_unet"
#mkdir "logs/$experiment_name"
#cd ./models
#python EUNet.py \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --ecg True \
#    --batch_size 128 \
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
#    --mix_pretraining_subject_samples True \
#    --max_training_epochs 1 \
#    --lr_scheduler_enable True \
#    --optimizer_type "Adam" \
#    --lr_scheduler_type "ExponentialLR" \
#    --criterion "SmoothL1Loss" \
#    --eval_every_n_epochs 1 \
#    --lambda_supervised 1 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"


## SSL UNet 
experiment_name="ssl_eunet_rand_masking"
mkdir "logs/$experiment_name"
cd ./models
python SSLEUNet.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --batch_size 64 \
    --channels "16,32,64" \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining_2.py \
    --model models.SSLEUNet \
    --dataset_name mimic_iii \
    --lp_dataset_name mimic_iii \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --sig2sig True \
    --ecg True \
    --ssl True \
    --fs 125 \
    --input_seq_len_s 5 \
    --channels "16,32,64" \
    --batch_size 64 \
    --mix_pretraining_subject_samples False \
    --max_training_epochs 100 \
    --max_lp_training_epochs 50 \
    --lr_scheduler_enable True \
    --optimizer_type "Adam" \
    --lr_scheduler_type "ExponentialLR" \
    --criterion "SmoothL1Loss" \
    --eval_every_n_epochs 1 \
    --apply_masking True \
    --lambda_msr 1 \
    --masking_ratio 0.15 \
    > "./logs/$experiment_name/${experiment_name}_training.log"