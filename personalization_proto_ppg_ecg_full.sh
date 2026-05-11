# Personalization

# PPG + ECG full model adaptation
experiment_name="personalization_proto_ppg_ecg_full_calibration_size_1"
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
    --pretrained_model_ckpt_path ./checkpoints/full_proto_ppg_ecg_percentile_group_layer_norm_kq_4/full_proto_ppg_ecg_percentile_group_layer_norm_kq_4-Proto-2026_05_05-12_50_57/full_proto_ppg_ecg_percentile_group_layer_norm_kq_4_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --calibration_phase_size 1 \
    --num_batches 76 \
    --num_blocks 1 \
    --replay_buffer_size 64 \
    --inner_adapt 'all' \
    --plot_personalization \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


experiment_name="personalization_proto_ppg_ecg_full_calibration_size_2"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --ecg \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 8 \
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
    --pretrained_model_ckpt_path ./checkpoints/full_proto_ppg_ecg_percentile_group_layer_norm_kq_8/full_proto_ppg_ecg_percentile_group_layer_norm_kq_8-Proto-2026_05_06-00_22_19/full_proto_ppg_ecg_percentile_group_layer_norm_kq_8_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --calibration_phase_size 2 \
    --num_batches 76 \
    --num_blocks 1 \
    --replay_buffer_size 64 \
    --inner_adapt 'all' \
    --plot_personalization \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"


experiment_name="personalization_proto_ppg_ecg_full_calibration_size_4"
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
    --pretrained_model_ckpt_path ./checkpoints/full_proto_ppg_ecg_percentile_group_layer_norm_kq_16/full_proto_ppg_ecg_percentile_group_layer_norm_kq_16-Proto-2026_05_06-05_23_52/full_proto_ppg_ecg_percentile_group_layer_norm_kq_16_best_maml \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --calibration_phase_size 4 \
    --num_batches 76 \
    --num_blocks 1 \
    --replay_buffer_size 64 \
    --inner_adapt 'all' \
    --plot_personalization \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"
