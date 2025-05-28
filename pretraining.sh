# Efficient UNet 
experiment_name="eunet"
mkdir "logs/$experiment_name"
cd ./models
python EUNet.py \
    --fs 125 \
    --input_seq_len_s 5 \
    --ecg True \
    --num_heads_attention 1 \
    --dim_feedforward_attention 128 \
    --return_embedding False \
    --set_tunable_params all \
    --batch_size 128 \
    --channels "16,32,63,128,256"
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
    --num_heads_attention 1 \
    --dim_feedforward_attention 128 \
    --channels "16,32,63,128,256" \
    --return_embedding False \
    --set_tunable_params all \
    --mix_pretraining_subject_samples True \
    --max_training_epochs 10 \
    --lr_scheduler_enable True \
    --optimizer_type "Adam" \
    --lr_scheduler_type "ExponentialLR" \
    --criterion "SmoothL1Loss" \
    --eval_every_n_epochs 1 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    --aug False \
    --aug False \
    > "./logs/$experiment_name/${experiment_name}_training.log"
