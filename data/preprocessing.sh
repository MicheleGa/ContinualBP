# MIMIC III Preprocessing
python mimic_iii_preprocessing.py --name mimic_iii_biot --num_threads 10 --fs 125 --window_length 10 --window_overlap 5 --percentile True > ./data_logs/mimic_iii_preprocessing.log
python mimic_iii_preprocessing.py --name mimic_iii_biot --num_threads 1 --fs 125 --window_length 10 --window_overlap 5 --percentile True --plot True
python dataset.py --name mimic_iii_biot --fs 125 --input_seq_len_s 10 --ecg True --sig2sig False --plot True --mix_pretraining_subject_samples False --loader_worker 10

#python mimic_iii_xl_preprocessing.py --name mimic_iii_xl_filtered --num_threads 1 --sig2sig True --rescale_to_unit True --window_length 10 --window_overlap 5 --butterworth_filter True --fir_bp_filtering True --plot True

#python pulse_db_preprocessing.py > ./data_logs/pulse_db_preprocessing.log
#python dataset.py --name mimic_iii_pulse_db --fs 125 --input_seq_len_s 10 --ecg True --sig2sig False --plot True --mix_pretraining_subject_samples False --loader_worker 10

# Vital DB Preprocessing
#python vital_db_preprocessing.py --name vital_db --num_threads 10 --fs 125 --window_length 10 --window_overlap 5 --percentile True > ./data_logs/vital_db.log
#python vital_db_preprocessing.py --name vital_db --num_threads 1 --fs 125 --window_length 10 --window_overlap 5 --percentile True --plot True 
#python online_dataset.py --name vital_db --ecg True --sig2sig False --input_seq_len_s 10 --fs 125 --plot True --min_run_length 200 --batch_size 32 --plot True
