# Personalization

experiment_name="biot_ppg_percentile"
mkdir "logs/$experiment_name"
cd ./models
python BIOT.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 1 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.BIOT \
    --dataset_name pulse_db_vital_db_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/biot_ppg_percentile/biot_ppg_percentile-BIOT-2025_12_01-18_09_54/biot_ppg_percentile_best_maml \
    --pretraining_feats_stats ./checkpoints/biot_ppg_percentile/biot_ppg_percentile-BIOT-2025_12_01-18_09_54/biot_ppg_percentile_embedding_stats.npz \
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
