# Vanilla Resnet w/ PPG + ECG 
experiment_name="personalization_proj_head_unfreeze_vanilla_resnet"
mkdir "logs/$experiment_name"
cd ./models
python ResGRUNet.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.ResGRUNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --pretrained_model_checkpoint ./checkpoints/vanilla_resnet/vanilla_resnet-ResGRUNet-2025_04_12-15_12_47/vanilla_resnet/ckpt/ResGRUNet \
    --loader_worker 4 \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params projection_head \
    --mix_pretraining_subject_samples True \
    --max_training_epochs 1000 \
    --optimizer_type "AdamW" \
    --lr 0.01 \
    --eval_every_n_epochs 1 \
    --batchsize 32 \
    --es_enable True \
    --es_patience 20 \
    --es_min_delta 0.005 \
    --personalization_sample_number 100 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    --aug False \
    > "./logs/$experiment_name/${experiment_name}_training.log"

# Vanilla Resnet w/ PPG + ECG without mixing subjects during pretraining
experiment_name="personalization_proj_head_unfreeze_vanilla_resnet_no_mix"
mkdir "logs/$experiment_name"
cd ./models
python ResGRUNet.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.ResGRUNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --pretrained_model_checkpoint ./checkpoints/vanilla_resnet_no_mix/vanilla_resnet_no_mix-ResGRUNet-2025_04_12-17_49_51/vanilla_resnet_no_mix/ckpt/ResGRUNet \
    --loader_worker 4 \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params projection_head \
    --mix_pretraining_subject_samples False \
    --max_training_epochs 1000 \
    --optimizer_type "AdamW" \
    --lr 0.01 \
    --eval_every_n_epochs 1 \
    --batchsize 32 \
    --es_enable True \
    --es_patience 20 \
    --es_min_delta 0.005 \
    --personalization_sample_number 100 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    --aug False \
    > "./logs/$experiment_name/${experiment_name}_training.log"

# Vanilla Resnet w/ PPG + ECG without mixing subjects during pretraining, smaller gradient
experiment_name="personalization_proj_head_unfreeze_vanilla_resnet_no_mix_lambda_supervised_0.0001"
mkdir "logs/$experiment_name"
cd ./models
python ResGRUNet.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.ResGRUNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --pretrained_model_checkpoint ./checkpoints/vanilla_resnet_no_mix_lambda_supervised_0.0001/vanilla_resnet_no_mix_lambda_supervised_0.0001-ResGRUNet-2025_04_12-20_27_02/vanilla_resnet_no_mix_lambda_supervised_0.0001/ckpt/ResGRUNet \
    --loader_worker 4 \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params projection_head \
    --mix_pretraining_subject_samples False \
    --max_training_epochs 1000 \
    --optimizer_type "AdamW" \
    --lr 0.01 \
    --eval_every_n_epochs 1 \
    --batchsize 32 \
    --es_enable True \
    --es_patience 20 \
    --es_min_delta 0.005 \
    --personalization_sample_number 100 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    --aug False \
    > "./logs/$experiment_name/${experiment_name}_training.log"

# Vanilla Resnet w/ PPG + ECG without mixing subjects during pretraining, smaller gradient ortho
experiment_name="personalization_proj_head_unfreeze_vanilla_resnet_no_mix_lambda_supervised_0.0001_ortho"
mkdir "logs/$experiment_name"
cd ./models
python ResGRUNet.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.ResGRUNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --pretrained_model_checkpoint ./checkpoints/vanilla_resnet_no_mix_lambda_supervised_0.0001_ortho/vanilla_resnet_no_mix_lambda_supervised_0.0001_ortho-ResGRUNet-2025_04_12-23_05_01/vanilla_resnet_no_mix_lambda_supervised_0.0001_ortho/ckpt/ResGRUNet \
    --loader_worker 4 \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params projection_head \
    --mix_pretraining_subject_samples False \
    --max_training_epochs 1000 \
    --optimizer_type "AdamW" \
    --lr 0.01 \
    --eval_every_n_epochs 1 \
    --batchsize 32 \
    --es_enable True \
    --es_patience 20 \
    --es_min_delta 0.005 \
    --personalization_sample_number 100 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    --aug False \
    > "./logs/$experiment_name/${experiment_name}_training.log"

# Vanilla Resnet w/ PPG + ECG without mixing subjects during pretraining, contrastive
experiment_name="personalization_proj_head_unfreeze_vanilla_resnet_no_mix_contrastive"
mkdir "logs/$experiment_name"
cd ./models
python ResGRUNet.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.ResGRUNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --pretrained_model_checkpoint ./checkpoints/vanilla_resnet_no_mix_contrastive/vanilla_resnet_no_mix_contrastive-ResGRUNet-2025_04_13-08_43_01/vanilla_resnet_no_mix_contrastive/ckpt/ResGRUNet \
    --loader_worker 4 \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params projection_head \
    --mix_pretraining_subject_samples False \
    --max_training_epochs 1000 \
    --optimizer_type "AdamW" \
    --lr 0.01 \
    --eval_every_n_epochs 1 \
    --batchsize 32 \
    --es_enable True \
    --es_patience 20 \
    --es_min_delta 0.005 \
    --personalization_sample_number 100 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    --aug False \
    > "./logs/$experiment_name/${experiment_name}_training.log"


