# MIMIC III Preprocessing
#python mimic_iii_preprocessing.py --name mimic_iii_biot --num_threads 10 --fs 125 --window_length 10 --window_overlap 5 --percentile True > ./data_logs/mimic_iii_preprocessing.log
#python mimic_iii_preprocessing.py --name mimic_iii_biot --num_threads 1 --fs 125 --window_length 10 --window_overlap 5 --percentile True --plot True
#python dataset.py --name mimic_iii_biot --fs 125 --input_seq_len_s 10 --ecg True --sig2sig False --plot True --mix_pretraining_subject_samples False --loader_worker 10

#python pulse_db_preprocessing.py > ./data_logs/pulse_db_preprocessing.log
#python dataset.py --name mimic_iii_pulse_db --fs 125 --input_seq_len_s 10 --ecg True --sig2sig False --plot True --mix_pretraining_subject_samples False --loader_worker 10

# Vital DB Preprocessing
python online_dataset.py --name vital_db_pulse_db --ecg False --sig2sig True --input_seq_len_s 10 --fs 125 --plot True --min_run_length 128 --batch_size 32 --plot True
