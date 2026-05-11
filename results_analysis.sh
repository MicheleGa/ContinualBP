# Results Analysis

# Main baseline
python results_analysis.py \
    --log_file_path "./logs/new_eval_gradual_shifts_ppg_batch_size_4/new_eval_gradual_shifts_ppg_batch_size_4_personalization_training.log" \
    --config_yaml_path "./checkpoints/full_proto_ppg_percentile_group_layer_norm_kq_16/full_proto_ppg_percentile_group_layer_norm_kq_16-Proto-2026_05_05-21_49_45/full_proto_ppg_percentile_group_layer_norm_kq_16/ckpt/config.yaml" \
    --fig_root "./figs/new_eval_gradual_shifts_ppg_batch_size_4" \
    --exp_fig_root "./figs/new_eval_gradual_shifts_ppg_batch_size_4/new_eval_gradual_shifts_ppg_batch_size_4-Proto-2026_05_08-12_20_02" \
    --exp_fig_root_drift_aware "./figs/new_eval_gradual_shifts_ppg_batch_size_4_drift_aware/new_eval_gradual_shifts_ppg_batch_size_4_drift_aware-Proto-2026_05_08-12_20_07"