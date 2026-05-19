# Personalization

# PPG + ECG w/ feature replay 
experiment_name="personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --ecg \
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
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_ecg_percentile_group_layer_norm_kq_4/proto_ppg_ecg_percentile_group_layer_norm_kq_4-Proto-2026_05_17-13_33_21/proto_ppg_ecg_percentile_group_layer_norm_kq_4_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 72 \
    --num_blocks 1 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'fixed' \
    --baselines "feature_replay" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


experiment_name="personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts_seed_40"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --ecg \
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
    --seed 40 \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_ecg_percentile_group_layer_norm_kq_4_seed_40/proto_ppg_ecg_percentile_group_layer_norm_kq_4_seed_40-Proto-2026_05_17-19_11_07/proto_ppg_ecg_percentile_group_layer_norm_kq_4_seed_40_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 72 \
    --num_blocks 1 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'fixed' \
    --baselines "feature_replay" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


experiment_name="personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts_seed_41"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --ecg \
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
    --seed 41 \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_ecg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_ecg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-22_50_23/proto_ppg_ecg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 72 \
    --num_blocks 1 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'fixed' \
    --baselines "feature_replay" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"