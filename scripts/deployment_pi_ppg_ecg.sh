# Drift Aware Personalization ~ Deployment on Raspberry Pi 5
# First move to main directory to make python scripts work with the right dependency paths
cd ..

# Always-on Baseline
experiment_name="pi_deployment_ppg_ecg_feature_replay_always_on"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 32 \
    --batch_size 4 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python deployment_personalization.py \
    --model models.Proto \
    --dataset_name pulse_db_vital_db_percentile \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 32 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4-Proto-2026_08_10-09_46_20/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 72 \
    --num_blocks 1 \
    --replay_buffer_size 16 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --calibration_phase_size 16 \
    --baselines "feature_replay" \
    --use_pi \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

# MMD Detector
experiment_name="pi_deployment_ppg_ecg_feature_replay_drift_aware_mmd"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 32 \
    --batch_size 4 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python deployment_personalization.py \
    --model models.Proto \
    --dataset_name pulse_db_vital_db_percentile \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 32 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4-Proto-2026_08_10-09_46_20/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 72 \
    --num_blocks 1 \
    --replay_buffer_size 16 \
    --inner_adapt 'head' \
    --setup_type 'drift' \
    --drift_detector_type 'mmd' \
    --calibration_phase_size 16 \
    --detector_window_size 4 \
    --detector_ert 64 \
    --detector_n_bootstraps 500 \
    --baselines "feature_replay" \
    --use_pi \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"