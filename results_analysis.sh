# Results Analysis

# Main baseline
python results_analysis.py \
    --log_file_path "./logs/gradual_shifts/gradual_shifts_personalization_training.log" \
    --config_yaml_path "./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm/ckpt/config.yaml" \
    --fig_root "./figs/gradual_shifts" \
    --exp_fig_root "./figs/gradual_shifts/gradual_shifts-Proto-2026_01_07-14_55_02" \
    --exp_fig_root_drift_aware "./figs/feature_drift_aware_gradual_shifts_0.1/feature_drift_aware_gradual_shifts_0.1-Proto-2026_03_09-11_44_48" "./figs/feature_drift_aware_gradual_shifts_0.2/feature_drift_aware_gradual_shifts_0.2-Proto-2026_03_09-11_57_47" "./figs/feature_drift_aware_gradual_shifts_0.3/feature_drift_aware_gradual_shifts_0.3-Proto-2026_03_09-12_10_01" "./figs/feature_drift_aware_gradual_shifts_0.4/feature_drift_aware_gradual_shifts_0.4-Proto-2026_03_09-12_22_18" "./figs/feature_drift_aware_gradual_shifts_0.5/feature_drift_aware_gradual_shifts_0.5-Proto-2026_03_09-12_34_23"