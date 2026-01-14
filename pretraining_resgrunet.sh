# Pretraining

# ResGruNet w/ Supervised + MAML Pre-training

# PPG percentile
#experiment_name="resgrunet_ppg_percentile"
#mkdir "logs/$experiment_name"
#cd ./models
#python ResGruNet.py \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --batch_size 16 \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python pretraining.py \
#    --model models.ResGruNet \
#    --dataset_name pulse_db_mimic_iii_percentile \
#    --expname "$experiment_name" \
#    --loader_worker 10 \
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
#    --k_support 16 \
#    --k_query 16 \
#    --meta_batch_size 4 \
#    --meta_lr_schedule 'cosine' \
#    --meta_lr 0.001 \
#    --meta_lr_scheduler_eta_min 0.00001 \
#    --inner_adapt 'head' \
#    --inner_opt 'adam' \
#    --inner_lr_schedule 'constant' \
#    --inner_lr 0.01 \
#    --inner_steps_schedule 'constant' \
#    --inner_steps 8 \
#    --eval_lr 0.01 \
#    --eval_steps 8 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"

# PPG z-score
#experiment_name="resgrunet_ppg_z_score"
#mkdir "logs/$experiment_name"
#cd ./models
#python ResGruNet.py \
#    --fs 125 \
#    --input_seq_len_s 10 \
#    --embed_dim 128 \
#    --batch_size 16 \
#    > "../logs/$experiment_name/${experiment_name}_summary.log"
#cd ..
#python pretraining.py \
#    --model models.ResGruNet \
#    --dataset_name pulse_db_mimic_iii_z_score \
#    --expname "$experiment_name" \
#    --loader_worker 10 \
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
#    --k_support 16 \
#    --k_query 16 \
#    --meta_batch_size 4 \
#    --meta_lr_schedule 'cosine' \
#    --meta_lr 0.001 \
#    --meta_lr_scheduler_eta_min 0.00001 \
#    --inner_adapt 'head' \
#    --inner_opt 'adam' \
#    --inner_lr_schedule 'constant' \
#    --inner_lr 0.01 \
#    --inner_steps_schedule 'constant' \
#    --inner_steps 8 \
#    --eval_lr 0.01 \
#    --eval_steps 8 \
#    > "./logs/$experiment_name/${experiment_name}_training.log"

experiment_name="resgrunet_ppg_min_max"
mkdir "logs/$experiment_name"
cd ./models
python ResGruNet.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 16 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.ResGruNet \
    --dataset_name pulse_db_mimic_iii_min_max \
    --expname "$experiment_name" \
    --loader_worker 10 \
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
    --inner_lr 0.01 \
    --inner_steps_schedule 'constant' \
    --inner_steps 8 \
    --eval_lr 0.01 \
    --eval_steps 8 \
    > "./logs/$experiment_name/${experiment_name}_training.log"