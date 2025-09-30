# Pretraining

# ResGruNet w/ Supervised + MAML Pre-training

# PPG
experiment_name="resgrunet_ppg_no_sig2sig"
mkdir "logs/$experiment_name"
cd ./models
python ResGruNet.py \
    --ecg False \
    --sig2sig False \
    --fs 125 \
    --input_seq_len_s 10 \
    --channels '1, 64, 128, 256' \
    --kernel_size 7 \
    --act 'leaky_relu' \
    --pooling 'avg' \
    --embed_dim 256 \
    --num_groups 8 \
    --batch_size 128 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.ResGruNet \
    --dataset_name mimic_iii_pulse_db \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --ecg False \
    --sig2sig False \
    --fs 125 \
    --input_seq_len_s 10 \
    --channels '1, 64, 128, 256' \
    --kernel_size 7 \
    --act 'leaky_relu' \
    --pooling 'avg' \
    --embed_dim 256 \
    --num_groups 8 \
    --batch_size 128 \
    --mix_pretraining_subject_samples False \
    --min_subject_sample_number 300 \
    --stage1_pre_train_lr 0.001 \
    --stage1_pre_train_scheduler_eta_min 0.00001 \
    --weight_decay 0.0001 \
    --stage1_epochs 20 \
    --criterion "SmoothL1Loss" \
    --max_meta_epochs 200 \
    --grad_clip 10.0 \
    --k_support 16 \
    --k_query 32 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --msl_include_pre False \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_head_lr_mult 1.0 \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.01 \
    --inner_steps_schedule 'constant' \
    --inner_steps 8 \
    --eval_lr 0.01 \
    --eval_steps 8 \
    > "./logs/$experiment_name/${experiment_name}_training.log"

# PPG + ECG
experiment_name="resgrunet_ppg_ecg_no_sig2sig"
mkdir "logs/$experiment_name"
cd ./models
python ResGruNet.py \
    --ecg True \
    --sig2sig False \
    --fs 125 \
    --input_seq_len_s 10 \
    --channels '1, 64, 128, 256' \
    --kernel_size 7 \
    --act 'leaky_relu' \
    --pooling 'avg' \
    --embed_dim 256 \
    --num_groups 8 \
    --batch_size 128 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.ResGruNet \
    --dataset_name mimic_iii_pulse_db \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --ecg True \
    --sig2sig False \
    --fs 125 \
    --input_seq_len_s 10 \
    --channels '1, 64, 128, 256' \
    --kernel_size 7 \
    --act 'leaky_relu' \
    --pooling 'avg' \
    --embed_dim 256 \
    --num_groups 8 \
    --batch_size 128 \
    --mix_pretraining_subject_samples False \
    --min_subject_sample_number 300 \
    --stage1_pre_train_lr 0.001 \
    --stage1_pre_train_scheduler_eta_min 0.00001 \
    --weight_decay 0.0001 \
    --stage1_epochs 20 \
    --criterion "SmoothL1Loss" \
    --max_meta_epochs 200 \
    --grad_clip 10.0 \
    --k_support 16 \
    --k_query 32 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --msl_include_pre False \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_head_lr_mult 1.0 \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.01 \
    --inner_steps_schedule 'constant' \
    --inner_steps 8 \
    --eval_lr 0.01 \
    --eval_steps 8 \
    > "./logs/$experiment_name/${experiment_name}_training.log"