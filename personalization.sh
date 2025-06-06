## Personalization

# Efficient UNet
experiment_name="personalization_eunet_last_layer"
mkdir "logs/$experiment_name"
cd ./models
python EUNet.py \
    --fs 125 \
    --input_seq_len_s 5 \
    --ecg True \
    --batch_size 128 \
    --channels "16,32,64" \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.EUNet \
    --dataset_name vital_db \
    --expname "$experiment_name" \
    --pretrained_model_checkpoint ./checkpoints/efficient_unet_ch_16_32_64/efficient_unet_ch_16_32_64-EUNet-2025_05_30-11_21_42/efficient_unet_ch_16_32_64/ckpt/EUNet \
    --loader_worker 4 \
    --batch_size 128 \
    --tune 'last_layer' \
    --channels "16,32,64" \
    --num_personalization_subjects 100 \
    --num_passes 8 \
    --lr 0.003 \
    --sig2sig True \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --mix_pretraining_subject_samples True \
    --personalization_sample_number -1.0 \
    --optimizer_type "Adam" \
    --criterion "SmoothL1Loss" \
    --eval_every_n_epochs 1 \
    --lambda_supervised 1 \
    > "./logs/$experiment_name/${experiment_name}_training.log"

## Efficient UNet Nystrom
#experiment_name="personalization_vanilla_eunet_nystrom_attention"
#mkdir "logs/$experiment_name"
#cd ./models
#python EUNet.py \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --ecg True \
#    --channels "16,32,64" \
#    --attention_type "nystrom_attention" \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python personalization.py \
#    --model models.EUNet \
#    --dataset_name vital_db \
#    --expname "$experiment_name" \
#    --pretrained_model_checkpoint ./checkpoints/efficient_unet_ch_16_32_64_nystrom/efficient_unet_ch_16_32_64_nystrom-EUNet-2025_05_30-11_22_04/efficient_unet_ch_16_32_64_nystrom/ckpt/EUNet \
#    --loader_worker 4 \
#    --sig2sig True \
#    --ecg True \
#    --fs 125 \
#    --input_seq_len_s 5 \
#    --channels "16,32,64" \
#    --attention_type "nystrom_attention" \
#    --mix_pretraining_subject_samples True \
#    --optimizer_type "Adam" \
#    --criterion "SmoothL1Loss" \
#    --eval_every_n_epochs 1 \
#    --lambda_supervised 1 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"
#