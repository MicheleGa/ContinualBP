# Personalization

# Ablation Studies on Normalization Methods
experiment_name="proto_ppg_z_score_group_layer_norm"
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
    --dataset_name pulse_db_vital_db_z_score \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_z_score_group_layer_norm/proto_ppg_z_score_group_layer_norm-Proto-2026_01_06-11_06_59/proto_ppg_z_score_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_z_score_group_layer_norm/proto_ppg_z_score_group_layer_norm-Proto-2026_01_06-11_06_59/proto_ppg_z_score_group_layer_norm_embedding_stats.npz \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 3 \
    --num_train_val 3 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="proto_ppg_min_max_group_layer_norm"
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
    --dataset_name pulse_db_vital_db_min_max \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_min_max_group_layer_norm/proto_ppg_min_max_group_layer_norm-Proto-2026_01_06-11_07_48/proto_ppg_min_max_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_min_max_group_layer_norm/proto_ppg_min_max_group_layer_norm-Proto-2026_01_06-11_07_48/proto_ppg_min_max_group_layer_norm_embedding_stats.npz \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 3 \
    --num_train_val 3 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"