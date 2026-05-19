# Personalization

# PPG head only adaptation
experiment_name="personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_41"
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
    --seed 41 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 83 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --baselines "no_adapt" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_41"
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
    --seed 41 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 83 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --baselines "first_batch_finetune" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_41"
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
    --seed 41 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 83 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --baselines "online" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_41"
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
    --seed 41 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 83 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --baselines "online_from_scratch" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

experiment_name="personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_41"
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
    --seed 41 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 83 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --baselines "feature_replay" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


experiment_name="personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_41"
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
    --seed 41 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 83 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --baselines "lwf" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


experiment_name="personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_41"
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
    --seed 41 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 83 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --baselines "ewc" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


experiment_name="personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_41"
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
    --seed 41 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4_seed_41/proto_ppg_percentile_group_layer_norm_kq_4_seed_41-Proto-2026_05_17-12_44_45/proto_ppg_percentile_group_layer_norm_kq_4_seed_41_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 83 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --setup_type 'fixed' \
    --baselines "agem" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"