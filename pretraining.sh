# Pretraining

# ResGruNet w/ Supervised + MAML Pre-training

# PPG
experiment_name="resgrunet_ppg_no_sig2sig_test"
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
    --num_groups 8 \
    --embed_dim 256 \
    --batch_size 128 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.ResGruNet \
    --dataset_name mimic_iii_pulse_db \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --sig2sig False \
    --ecg False \
    --fs 125 \
    --input_seq_len_s 10 \
    --channels '1, 64, 128, 256' \
    --kernel_size 7 \
    --act 'leaky_relu' \
    --pooling 'avg' \
    --num_groups 8 \
    --embed_dim 256 \
    --batch_size 128 \
    --mix_pretraining_subject_samples False \
    --min_subject_sample_number 300 \
    --stage1_pre_train_lr 0.001 \
    --stage1_pre_train_scheduler_eta_min 0.00001 \
    --weight_decay 0.0001 \
    --stage1_epochs 20 \
    --criterion "SmoothL1Loss" \
    --max_meta_epochs 150 \
    --grad_clip 10.0 \
    --k_support 32 \
    --k_query 64 \
    --meta_batch_size 4 \
    --use_pure_functional True \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --msl_include_pre False \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_head_lr_mult 1.0 \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.005 \
    --inner_steps_schedule 'constant' \
    --inner_steps 5 \
    --eval_lr 0.005 \
    --eval_steps 5 \
    > "./logs/$experiment_name/${experiment_name}_training.log"