# Drift Detection Calibration After Pretraining

# PPG only with frozen backbone
#experiment_name="mmd_drift_detector_calibration_proto_ppg_kq_16"
#mkdir "logs/$experiment_name"
#cd ./models
#python Proto.py \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --batch_size 16 \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python drift_detector_calibration.py \
#    --model models.Proto \
#    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_16/proto_ppg_percentile_group_layer_norm_kq_16-Proto-2026_05_04-17_16_13/proto_ppg_percentile_group_layer_norm_kq_16_best_maml \
#    --detector_calibration_csv_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_16/proto_ppg_percentile_group_layer_norm_kq_16-Proto-2026_05_04-17_16_13 \
#    --dataset_name pulse_db_mimic_iii_percentile \
#    --expname "$experiment_name" \
#    --loader_worker 10 \
#    --gpu 0 \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --criterion 'SmoothL1Loss' \
#    --inner_adapt 'head' \
#    --personalization_lr 0.005 \
#    --personalization_steps 10 \
#    --personalization_batch_size 4 \
#    --num_batches 76 \
#    --num_blocks 1 \
#    --calibration_phase_size 4 \
#    --drift_detector_type 'mmd' \
#    --setup_type "drift" \
#    > "./logs/$experiment_name/${experiment_name}_training.log"

experiment_name="lsdd_drift_detector_calibration_proto_ppg_kq_16"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 16 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python drift_detector_calibration.py \
    --model models.Proto \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_16/proto_ppg_percentile_group_layer_norm_kq_16-Proto-2026_05_04-17_16_13/proto_ppg_percentile_group_layer_norm_kq_16_best_maml \
    --detector_calibration_csv_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_16/proto_ppg_percentile_group_layer_norm_kq_16-Proto-2026_05_04-17_16_13 \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --criterion 'SmoothL1Loss' \
    --inner_adapt 'head' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --num_batches 76 \
    --num_blocks 1 \
    --calibration_phase_size 4 \
    --drift_detector_type 'lsdd' \
    --setup_type "drift" \
    > "./logs/$experiment_name/${experiment_name}_training.log"
