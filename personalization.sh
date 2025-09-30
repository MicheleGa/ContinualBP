# Personalization

# PPG no sig2sig
experiment_name="resgrunet_ppg_no_sig2sig_personalization"
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
python personalization.py \
    --model models.ResGruNet \
    --dataset_name vital_db_pulse_db \
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
    --num_groups 8 \
    --embed_dim 256 \
    --pretrained_model_ckpt_path ./checkpoints/resgrunet_ppg_no_sig2sig/resgrunet_ppg_no_sig2sig-ResGruNet-2025_09_28-18_32_25/resgrunet_ppg_no_sig2sig_best_maml \
    --pretraining_feats_stats ./checkpoints/resgrunet_ppg_no_sig2sig/resgrunet_ppg_no_sig2sig-ResGruNet-2025_09_28-18_32_25/resgrunet_ppg_no_sig2sig_embedding_stats.npz \
    --min_run_length 128 \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --grad_clip 10.0 \
    --inner_adapt 'head' \
    --inner_head_lr_mult 1.0 \
    --num_personalization_subjects 0 \
    --plot_personalization True \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

# PPG + ECG no sig2sig
experiment_name="resgrunet_ppg_ecg_no_sig2sig_personalization"
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
    --num_groups 8 \
    --embed_dim 256 \
    --batch_size 128 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.ResGruNet \
    --dataset_name vital_db_pulse_db \
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
    --num_groups 8 \
    --embed_dim 256 \
    --pretrained_model_ckpt_path ./checkpoints/resgrunet_ppg_ecg_no_sig2sig/resgrunet_ppg_ecg_no_sig2sig-ResGruNet-2025_09_29-03_47_46/resgrunet_ppg_ecg_no_sig2sig_best_maml \
    --pretraining_feats_stats ./checkpoints/resgrunet_ppg_ecg_no_sig2sig/resgrunet_ppg_ecg_no_sig2sig-ResGruNet-2025_09_29-03_47_46/resgrunet_ppg_ecg_no_sig2sig_embedding_stats.npz \
    --min_run_length 128 \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --grad_clip 10.0 \
    --inner_adapt 'head' \
    --inner_head_lr_mult 1.0 \
    --num_personalization_subjects 0 \
    --plot_personalization True \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

# PPG sig2sig
experiment_name="resgrunet_ppg_sig2sig_personalization"
mkdir "logs/$experiment_name"
cd ./models
python ResGruNet.py \
    --ecg False \
    --sig2sig True \
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
python personalization.py \
    --model models.ResGruNet \
    --dataset_name vital_db_pulse_db \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --ecg False \
    --sig2sig True \
    --fs 125 \
    --input_seq_len_s 10 \
    --channels '1, 64, 128, 256' \
    --kernel_size 7 \
    --act 'leaky_relu' \
    --pooling 'avg' \
    --num_groups 8 \
    --embed_dim 256 \
    --pretrained_model_ckpt_path ./checkpoints/resgrunet_ppg_sig2sig/resgrunet_ppg_sig2sig-ResGruNet-2025_09_28-18_32_33/resgrunet_ppg_sig2sig_best_maml \
    --pretraining_feats_stats ./checkpoints/resgrunet_ppg_sig2sig/resgrunet_ppg_sig2sig-ResGruNet-2025_09_28-18_32_33/resgrunet_ppg_sig2sig_embedding_stats.npz \
    --min_run_length 128 \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --grad_clip 10.0 \
    --inner_adapt 'head' \
    --inner_head_lr_mult 1.0 \
    --num_personalization_subjects 0 \
    --plot_personalization True \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"