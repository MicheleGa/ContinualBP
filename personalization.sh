# Personalization

# PPG Only
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
    --pretrained_model_ckpt_path ./checkpoints/resgrunet_ppg_no_sig2sig/resgrunet_ppg_no_sig2sig-ResGruNet-2025_09_21-01_14_40/resgrunet_ppg_no_sig2sig_best_maml \
    --pretraining_feats_stats ./checkpoints/resgrunet_ppg_no_sig2sig/resgrunet_ppg_no_sig2sig-ResGruNet-2025_09_21-01_14_40/resgrunet_ppg_no_sig2sig_embedding_stats.npz \
    --min_run_length 128 \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 5 \
    --personalization_batch_size 32 \
    --grad_clip 10.0 \
    --inner_adapt 'head' \
    --inner_head_lr_mult 1.0 \
    --num_personalization_subjects 0 \
    --plot_personalization True \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

# PPG Only sig2sig
experiment_name="resgrunet_ppg_personalization"
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
    --pretrained_model_ckpt_path ./checkpoints/resgrunet_ppg/resgrunet_ppg-ResGruNet-2025_09_19-09_06_59/resgrunet_ppg_best_maml \
    --pretraining_feats_stats ./checkpoints/resgrunet_ppg/resgrunet_ppg-ResGruNet-2025_09_19-09_06_59/resgrunet_ppg_embedding_stats.npz \
    --min_run_length 128 \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 5 \
    --personalization_batch_size 32 \
    --grad_clip 10.0 \
    --inner_adapt 'head' \
    --inner_head_lr_mult 1.0 \
    --num_personalization_subjects 0 \
    --plot_personalization True \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

# PPG + ECG
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
    --pretrained_model_ckpt_path ./checkpoints/resgrunet_ppg_ecg_no_sig2sig/resgrunet_ppg_ecg_no_sig2sig-ResGruNet-2025_09_21-10_36_42/resgrunet_ppg_ecg_no_sig2sig_best_maml \
    --pretraining_feats_stats ./checkpoints/resgrunet_ppg_ecg_no_sig2sig/resgrunet_ppg_ecg_no_sig2sig-ResGruNet-2025_09_21-10_36_42/resgrunet_ppg_ecg_no_sig2sig_embedding_stats.npz \
    --min_run_length 128 \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 5 \
    --personalization_batch_size 32 \
    --grad_clip 10.0 \
    --inner_adapt 'head' \
    --inner_head_lr_mult 1.0 \
    --num_personalization_subjects 0 \
    --plot_personalization True \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

# PPG + ECG sig2sig
experiment_name="resgrunet_ppg_ecg_personalization"
mkdir "logs/$experiment_name"
cd ./models
python ResGruNet.py \
    --ecg True \
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
    --ecg True \
    --sig2sig True \
    --fs 125 \
    --input_seq_len_s 10 \
    --channels '1, 64, 128, 256' \
    --kernel_size 7 \
    --act 'leaky_relu' \
    --pooling 'avg' \
    --num_groups 8 \
    --embed_dim 256 \
    --pretrained_model_ckpt_path ./checkpoints/resgrunet_ppg_ecg/resgrunet_ppg_ecg-ResGruNet-2025_09_19-23_10_58/resgrunet_ppg_ecg_best_maml \
    --pretraining_feats_stats ./checkpoints/resgrunet_ppg_ecg/resgrunet_ppg_ecg-ResGruNet-2025_09_19-23_10_58/resgrunet_ppg_ecg_embedding_stats.npz \
    --min_run_length 128 \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 5 \
    --personalization_batch_size 32 \
    --grad_clip 10.0 \
    --inner_adapt 'head' \
    --inner_head_lr_mult 1.0 \
    --num_personalization_subjects 0 \
    --plot_personalization True \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"
