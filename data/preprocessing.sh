# Preprocessing

# Plot Sample Signals
# PulseDB
#python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_percentile --num_threads 1 --plot --normalization percentile
#python pulse_db_preprocessing.py --input_folder ./pulse_db/vital_db_npz --index_file_name ./pulse_db/vital_db_index.csv --name pulse_db_vital_db_percentile --num_threads 1 --plot --normalization percentile
#
#python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_z_score --num_threads 1 --plot --normalization z_score
#python pulse_db_preprocessing.py --input_folder ./pulse_db/vital_db_npz --index_file_name ./pulse_db/vital_db_index.csv --name pulse_db_vital_db_z_score --num_threads 1 --plot --normalization z_score
#
#python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_min_max --num_threads 1 --plot --normalization min_max
#python pulse_db_preprocessing.py --input_folder ./pulse_db/vital_db_npz --index_file_name ./pulse_db/vital_db_index.csv --name pulse_db_vital_db_min_max --num_threads 1 --plot --normalization min_max

# AuoraDB
#python aurora_db_preprocessing.py --input_folder ./aurora_db --name aurora_db_percentile --num_threads 1 --plot

# Build lmdb
# PulseDB
# Percentile normalization
#python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_percentile --num_threads 8 --normalization percentile > ./data_logs/pulse_db_mimic_iii_percentile_preprocessing.log
#python pulse_db_preprocessing.py --input_folder ./pulse_db/vital_db_npz --index_file_name ./pulse_db/vital_db_index.csv --name pulse_db_vital_db_percentile --num_threads 8 --normalization percentile > ./data_logs/pulse_db_vital_db_percentile_preprocessing.log

# Z-score normalization
#python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_z_score --num_threads 8 --normalization z_score > ./data_logs/pulse_db_mimic_iii_z_score_preprocessing.log
#python pulse_db_preprocessing.py --input_folder ./pulse_db/vital_db_npz --index_file_name ./pulse_db/vital_db_index.csv --name pulse_db_vital_db_z_score --num_threads 8 --normalization z_score > ./data_logs/pulse_db_vital_db_z_score_preprocessing.log

# Min-Max scaling
#python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_min_max --num_threads 8 --normalization min_max > ./data_logs/pulse_db_mimic_iii_z_score_preprocessing.log
#python pulse_db_preprocessing.py --input_folder ./pulse_db/vital_db_npz --index_file_name ./pulse_db/vital_db_index.csv --name pulse_db_vital_db_min_max --num_threads 8 --normalization min_max > ./data_logs/pulse_db_vital_db_z_score_preprocessing.log

# AuroraDB
#python aurora_db_preprocessing.py --input_folder ./aurora_db --name aurora_db_percentile --num_threads 8 > ./data_logs/aurora_db_preprocessing.log

# Plot Statistics
#python dataset.py --name pulse_db_mimic_iii_percentile --fs 125 --input_seq_len_s 10 --plot --loader_worker 10 --index_file_name ./pulse_db/mimic_iii_index.csv --plot
#python dataset.py --name pulse_db_vital_db_percentile --fs 125 --input_seq_len_s 10 --plot --loader_worker 10 --index_file_name ./pulse_db/vital_db_index.csv --plot
#python dataset.py --name aurora_db_percentile --fs 125 --ecg --input_seq_len_s 10 --plot --loader_worker 10 --index_file_name ./aurora_db/aurora_db_index.csv 

# Meta-Learning
#python meta_dataloaders.py --dataset_name pulse_db_mimic_iii_percentile --k_support 16 --k_query 16 --plot --index_file_name ./pulse_db/mimic_iii_index.csv
#python meta_dataloaders.py --dataset_name aurora_db_percentile --fs 125 --ecg --k_support 16 --k_query 16 --plot --index_file_name ./aurora_db/aurora_db_index.csv

# Online Dataset
#python online_dataset.py --name pulse_db_vital_db_percentile --input_seq_len_s 10 --fs 125 --plot --personalization_batch_size 16 --num_batches 3 --num_blocks 4

python online_dataset_aurora.py --name aurora_db_percentile --input_seq_len_s 10 --fs 125 --plot --personalization_batch_size 16 --num_batches 1 --num_blocks 3
