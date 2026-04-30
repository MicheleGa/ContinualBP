# Ablation Studies on Drift Aware Personalization
# FeatureDriftDetector with Mahalanobis distance

experiment_name="feature_drift_aware_gradual_shifts_0.1"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 1 \
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
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/feature_stats.pt \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 1 \
    --num_train_val 9 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'drift' \
    --drift_threshold 0.1 \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="feature_drift_aware_gradual_shifts_0.2"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 1 \
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
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/feature_stats.pt \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 1 \
    --num_train_val 9 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'drift' \
    --drift_threshold 0.2 \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="feature_drift_aware_gradual_shifts_0.3"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 1 \
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
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/feature_stats.pt \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 1 \
    --num_train_val 9 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'drift' \
    --drift_threshold 0.3 \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="feature_drift_aware_gradual_shifts_0.4"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 1 \
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
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/feature_stats.pt \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 1 \
    --num_train_val 9 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'drift' \
    --drift_threshold 0.4 \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="feature_drift_aware_gradual_shifts_0.5"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 1 \
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
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/feature_stats.pt \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 1 \
    --num_train_val 9 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'drift' \
    --drift_threshold 0.5 \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"