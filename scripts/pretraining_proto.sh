# Pretraining
# First move to main directory to make python scripts work with the right dependency paths
cd ..

# Model: Proto -> CNN + GRU + MLP w/ group/layer norm
# Algorithm: Proto w/ Supervised + MAML Pre-training
# Dataset: PulseDB

# PPG percentile
experiment_name="proto_ppg_percentile_group_layer_norm_kq_4"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 4 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.Proto \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 4 \
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
    --k_support 4 \
    --k_query 4 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'cosine' \
    --inner_lr 0.01 \
    --inner_lr_min 0.005 \
    --inner_steps_schedule 'cosine' \
    --inner_steps 5 \
    --inner_steps_max 10 \
    --eval_lr 0.005 \
    --eval_steps 10 \
    > "./logs/$experiment_name/${experiment_name}_training.log"

#experiment_name="full_proto_ppg_percentile_group_layer_norm_kq_4"
#mkdir "logs/$experiment_name"
#cd ./models
#python Proto.py \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --batch_size 4 \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python pretraining.py \
#    --model models.Proto \
#    --dataset_name pulse_db_mimic_iii_percentile \
#    --expname "$experiment_name" \
#    --loader_worker 4 \
#    --gpu 0 \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --batch_size 128 \
#    --stage1_pre_train_lr 0.001 \
#    --stage1_pre_train_scheduler_eta_min 0.00001 \
#    --weight_decay 0.0001 \
#    --stage1_epochs 10 \
#    --criterion "SmoothL1Loss" \
#    --max_meta_epochs 200 \
#    --k_support 4 \
#    --k_query 4 \
#    --meta_batch_size 4 \
#    --meta_lr_schedule 'cosine' \
#    --meta_lr 0.001 \
#    --meta_lr_scheduler_eta_min 0.00001 \
#    --inner_adapt 'all' \
#    --inner_opt 'adam' \
#    --inner_lr_schedule 'cosine' \
#    --inner_lr 0.01 \
#    --inner_lr_min 0.005 \
#    --inner_steps_schedule 'cosine' \
#    --inner_steps 5 \
#    --inner_steps_max 10 \
#    --eval_lr 0.005 \
#    --eval_steps 10 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"

experiment_name="full_proto_ppg_percentile_group_layer_norm_kq_8"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 8 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.Proto \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 4 \
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
    --k_support 8 \
    --k_query 8 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --inner_adapt 'all' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'cosine' \
    --inner_lr 0.01 \
    --inner_lr_min 0.005 \
    --inner_steps_schedule 'cosine' \
    --inner_steps 5 \
    --inner_steps_max 10 \
    --eval_lr 0.005 \
    --eval_steps 10 \
    > "./logs/$experiment_name/${experiment_name}_training.log"

experiment_name="full_proto_ppg_percentile_group_layer_norm_kq_16"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 16 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.Proto \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 4 \
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
    --inner_adapt 'all' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'cosine' \
    --inner_lr 0.01 \
    --inner_lr_min 0.005 \
    --inner_steps_schedule 'cosine' \
    --inner_steps 5 \
    --inner_steps_max 10 \
    --eval_lr 0.005 \
    --eval_steps 10 \
    > "./logs/$experiment_name/${experiment_name}_training.log"