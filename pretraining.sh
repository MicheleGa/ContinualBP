# Pretraining

# Supervised Efficient UNet
experiment_name="efficient_unet"
mkdir "logs/$experiment_name"
cd ./models
python EUNet.py \
    --fs 125 \
    --input_seq_len_s 5 \
    --ecg True \
    --batch_size 128 \
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
    --mix_pretraining_subject_samples True \
    --max_training_epochs 1 \
    --lr_scheduler_enable True \
    --optimizer_type "Adam" \
    --lr_scheduler_type "ExponentialLR" \
    --criterion "SmoothL1Loss" \
    --eval_every_n_epochs 1 \
    --lambda_supervised 1 \
    > "./logs/$experiment_name/${experiment_name}_training.log"


## SSL UNet 
#experiment_name="ssl_unet"
#mkdir "logs/$experiment_name"
#cd ./models
#python SSLUNet.py \
#    --ecg True \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --proj_hidden_dim 1024 \
#    --proj_head_dim 256 \
#    --batch_size 128 \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python pretraining.py \
#    --model models.SSLUNet \
#    --dataset_name mimic_iii_ssl \
#    --lp_dataset_name mimic_iii \
#    --expname "$experiment_name" \
#    --loader_worker 4 \
#    --sig2sig True \
#    --ecg True \
#    --ssl True \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --proj_hidden_dim 1024 \
#    --proj_head_dim 256 \
#    --batch_size 128 \
#    --mix_pretraining_subject_samples True \
#    --max_training_epochs 1 \
#    --max_lp_training_epochs 1 \
#    --lr_scheduler_enable True \
#    --optimizer_type "Adam" \
#    --lr_scheduler_type "ExponentialLR" \
#    --criterion "SmoothL1Loss" \
#    --eval_every_n_epochs 1 \
#    --lambda_contrastive 0.5 \
#    --temperature 0.07 \
#    --lambda_cwg 0.25 \
#    --lambda_msr 0.25 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"