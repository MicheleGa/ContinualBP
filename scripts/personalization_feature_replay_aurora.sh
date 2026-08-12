# Personalization on Aurora DB
# First move to main directory to make python scripts work with the right dependency paths
cd ..

# PPG head only adaptation
experiment_name="personalization_feature_replay_proto_ppg_calibration_size_1_aurora"
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
    --dataset_name aurora_db_percentile \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4/proto_ppg_percentile_group_layer_norm_kq_4-Proto-2026_05_14-21_57_34/proto_ppg_percentile_group_layer_norm_kq_4_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 1 \
    --num_blocks 13 \
    --replay_buffer_size 64 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'fixed' \
    --baselines "feature_replay" \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"

