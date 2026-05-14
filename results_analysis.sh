# Results Analysis

# Main baseline
python results_analysis.py \
    --log_file_path "./logs/personalization_proto_ppg_calibration_size_4/personalization_proto_ppg_calibration_size_4_personalization_training.log" \
    --config_yaml_path "./checkpoints/proto_ppg_percentile_group_layer_norm_kq_16/proto_ppg_percentile_group_layer_norm_kq_16-Proto-2026_05_04-17_16_13/proto_ppg_percentile_group_layer_norm_kq_16/ckpt/config.yaml" \
    --fig_root "./figs/personalization_proto_ppg_calibration_size_4" \
    --exp_fig_root "./figs/personalization_proto_ppg_calibration_size_4/personalization_proto_ppg_calibration_size_4-Proto-2026_05_12-22_56_11" \
    --exp_pi_deployment_drift_aware_fig_root "./figs/pi_feature_replay_deployment_drift/pi_feature_replay_deployment_drift-Proto-2026_05_13-12_55_21" 