# Drift Detection Calibration After Pretraining
# First move to main directory to make python scripts work with the right dependency paths
cd ..

# PPG only with frozen backbone
experiment_name="lsdd_drift_detector_calibration_proto_ppg_kq_4"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 4 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python drift_detector_calibration.py \
    --model models.Proto \
    --detector_calibration_csv_path ./checkpoints/proto_ppg_percentile_group_layer_norm_kq_4/proto_ppg_percentile_group_layer_norm_kq_4-Proto-2026_05_14-21_57_34 \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --gpu 1 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --criterion 'SmoothL1Loss' \
    --inner_adapt 'head' \
    --personalization_lr 0.005 \
    --personalization_steps 10 \
    --personalization_batch_size 4 \
    --calibration_phase_size 1 \
    --num_batches 72 \
    --num_blocks 1 \
    --drift_detector_type 'lsdd' \
    --setup_type "drift" \
    > "./logs/$experiment_name/${experiment_name}_training.log"