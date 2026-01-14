# Results Analysis

# Main baseline
python results_analysis.py \
    --log_file_path "./logs/gradual_shifts/gradual_shifts_personalization_training.log" \
    --config_yaml_path "./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm/ckpt/config.yaml" \
    --fig_root "./figs/gradual_shifts" \
    --exp_fig_root "./figs/gradual_shifts/gradual_shifts-Proto-2026_01_07-14_55_02" \
    --exp_fig_root_drift_aware "./figs/drift_aware_gradual_shifts_th_5/drift_aware_gradual_shifts_th_5-Proto-2026_01_13-12_22_36"
