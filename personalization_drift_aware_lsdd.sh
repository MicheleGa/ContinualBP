# Drift Aware Personalization

# FeatureDriftDetector with Online LSDD w/ reinitialization
experiment_name="personalization_feature_replay_proto_ppg_drift_aware_lsdd"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 4 \
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
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4/proto_ppg_percentile_group_layer_norm_kq_4-Proto-2026_05_14-21_57_34/proto_ppg_percentile_group_layer_norm_kq_4_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 72 \
    --num_blocks 1 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'drift' \
    --drift_detector_type 'lsdd' \
    --calibration_phase_size 16 \
    --detector_window_size 4 \
    --detector_ert 32 \
    --detector_n_bootstraps 500 \
    --baselines "feature_replay" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


#experiment_name="personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_40"
#mkdir "logs/$experiment_name"
#cd ./models
#python Proto.py \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --batch_size 4 \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python personalization.py \
#    --model models.Proto \
#    --dataset_name pulse_db_vital_db_percentile \
#    --expname "$experiment_name" \
#    --loader_worker 10 \
#    --gpu 0 \
#    --seed 40 \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_40/proto_ppg_percentile_group_layer_norm_kq_4_seed_40-Proto-2026_05_17-10_53_46/proto_ppg_percentile_group_layer_norm_kq_4_seed_40_best_maml \
#    --criterion 'SmoothL1Loss' \
#    --personalization_lr 0.005 \
#    --personalization_steps 10 \
#    --personalization_batch_size 4 \
#    --num_batches 72 \
#    --num_blocks 1 \
#    --replay_buffer_size 64 \
#    --inner_adapt 'head' \
#    --plot_personalization \
#    --setup_type 'drift' \
#    --drift_detector_type 'lsdd' \
#    --calibration_phase_size 16 \
#    --detector_window_size 4 \
#    --detector_ert 64 \
#    --detector_n_bootstraps 500 \
#    --baselines "feature_replay" \
#    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


#experiment_name="personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_41"
#mkdir "logs/$experiment_name"
#cd ./models
#python Proto.py \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --batch_size 4 \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python personalization.py \
#    --model models.Proto \
#    --dataset_name pulse_db_vital_db_percentile \
#    --expname "$experiment_name" \
#    --loader_worker 10 \
#    --gpu 0 \
#    --seed 41 \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
#    --criterion 'SmoothL1Loss' \
#    --personalization_lr 0.005 \
#    --personalization_steps 10 \
#    --personalization_batch_size 4 \
#    --num_batches 72 \
#    --num_blocks 1 \
#    --replay_buffer_size 64 \
#    --inner_adapt 'head' \
#    --plot_personalization \
#    --setup_type 'drift' \
#    --drift_detector_type 'lsdd' \
#    --calibration_phase_size 16 \
#    --detector_window_size 4 \
#    --detector_ert 64 \
#    --detector_n_bootstraps 500 \
#    --baselines "feature_replay" \
#    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"
