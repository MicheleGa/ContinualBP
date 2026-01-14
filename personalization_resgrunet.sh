# Personalization

experiment_name="resgrunet_ppg_percentile"
mkdir "logs/$experiment_name"
cd ./models
python ResGruNet.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 16 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.ResGruNet \
    --dataset_name pulse_db_vital_db_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/resgrunet_ppg_percentile/resgrunet_ppg_percentile-ResGruNet-2025_12_03-09_11_00/resgrunet_ppg_percentile_best_maml \
    --pretraining_feats_stats ./checkpoints/resgrunet_ppg_percentile/resgrunet_ppg_percentile-ResGruNet-2025_12_03-09_11_00/resgrunet_ppg_percentile_embedding_stats.npz \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 5 \
    --num_train_val 2 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"