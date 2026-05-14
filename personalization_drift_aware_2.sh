# Drift Aware Personalization

# FeatureDriftDetector with Online LSDD w/ reinitialization
experiment_name="feature_replay_personalization_proto_ppg_calibration_size_4_drift_aware_online_lsdd"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 16 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.Proto \
    --dataset_name pulse_db_vital_db_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_16/proto_ppg_percentile_group_layer_norm_kq_16-Proto-2026_05_04-17_16_13/proto_ppg_percentile_group_layer_norm_kq_16_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --calibration_phase_size 4 \
    --num_batches 76 \
    --num_blocks 1 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'drift' \
    --drift_detector_type 'lsdd' \
    --detector_window_size 3 \
    --detector_ert 64 \
    --detector_n_bootstraps 500 \
    --baselines "feature_replay" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"