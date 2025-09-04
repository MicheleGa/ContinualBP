# Personalization

# BIOT personalization
experiment_name="biot_maml_personalization"
mkdir "logs/$experiment_name"
cd ./models
python BIOT.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 10 \
    --pretrained_path ../checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.BIOT \
    --dataset_name vital_db \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --sig2sig False \
    --fs 125 \
    --input_seq_len_s 10 \
    --pretrained_path ./checkpoints/pretrained_biot_encoder/EEG-six-datasets-18-channels.ckpt \
    --ecg True \
    --pretrained_model_checkpoint ./checkpoints/biot_maml/biot_maml-BIOT-2025_09_01-17_13_16/biot_maml_best_maml \
    --min_run_length 10 \
    --training_samples 8 \
    --criterion 'SmoothL1Loss' \
    --personalization_steps 5 \
    --grad_clip 10.0 \
    > "./logs/$experiment_name/${experiment_name}_personalization.log"