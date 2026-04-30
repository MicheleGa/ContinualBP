# Compute Statistics after pretraining

# Using Proto (group layer norm) w/ Supervised + MAML Pre-training + PPG percentile
experiment_name="compute_stats_proto_ppg_percentile_group_layer_norm"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 16 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python collect_stats_after_pretraining.py \
    --model models.Proto \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_04_29-14_42_06/proto_ppg_percentile_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_04_29-14_42_06/feature_stats.pt \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 128 \
    --stage1_pre_train_lr 0.001 \
    --stage1_pre_train_scheduler_eta_min 0.00001 \
    --weight_decay 0.0001 \
    --stage1_epochs 10 \
    --criterion "SmoothL1Loss" \
    --max_meta_epochs 200 \
    --k_support 16 \
    --k_query 16 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.005 \
    --inner_steps_schedule 'constant' \
    --inner_steps 10 \
    --eval_lr 0.005 \
    --eval_steps 10 \
    > "./logs/$experiment_name/${experiment_name}_training.log"

experiment_name="compute_stats_proto_ppg_ecg_percentile_group_layer_norm"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 16 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python collect_stats_after_pretraining.py \
    --model models.Proto \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_ecg_percentile_group_layer_norm/proto_ppg_ecg_percentile_group_layer_norm-Proto-2026_04_29-16_42_19/proto_ppg_ecg_percentile_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_ecg_percentile_group_layer_norm/proto_ppg_ecg_percentile_group_layer_norm-Proto-2026_04_29-16_42_19/feature_stats.pt \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 128 \
    --stage1_pre_train_lr 0.001 \
    --stage1_pre_train_scheduler_eta_min 0.00001 \
    --weight_decay 0.0001 \
    --stage1_epochs 10 \
    --criterion "SmoothL1Loss" \
    --max_meta_epochs 200 \
    --k_support 16 \
    --k_query 16 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.005 \
    --inner_steps_schedule 'constant' \
    --inner_steps 10 \
    --eval_lr 0.005 \
    --eval_steps 10 \
    > "./logs/$experiment_name/${experiment_name}_training.log"
