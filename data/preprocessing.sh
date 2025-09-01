# MIMIC III Preprocessing
python mimic_iii_preprocessing.py --name mimic_iii_biot --num_threads 10 --fs 125 --window_length 10 --window_overlap 5 --percentile True > ./data_logs/mimic_iii_preprocessing.log
python mimic_iii_preprocessing.py --name mimic_iii_biot --num_threads 1 --fs 125 --window_length 10 --window_overlap 5 --percentile True --plot True
python dataset.py --name mimic_iii_biot --fs 125 --input_seq_len_s 10 --ecg True --sig2sig False --plot True --mix_pretraining_subject_samples False --loader_worker 10


#python mimic_iii_xl_preprocessing.py --name mimic_iii_xl_filtered --num_threads 1 --sig2sig True --rescale_to_unit True --window_length 10 --window_overlap 5 --butterworth_filter True --fir_bp_filtering True --plot True

#python biot_mimic_iii_preprocessing.py --name biot_mimic_iii --num_threads 10 --fs 125 --target_fs 200 --token_size 200 --hop_length 100 --window_length 10 --ecg True --sig2sig True > ./data_logs/biot_mimic_iii_preprocessing.log
#python biot_mimic_iii_preprocessing.py --name biot_mimic_iii --num_threads 1 --fs 125 --target_fs 200 --token_size 200 --hop_length 100 --window_length 10 --ecg True --sig2sig True --plot True

#python biot_mimic_iii_preprocessing_2.py --name biot_mimic_iii --num_threads 10 --fs 125 --target_fs 200 --token_size 200 --hop_length 100 --window_length 10 --ecg True --sig2sig True > ./data_logs/biot_mimic_iii_preprocessing.log
#python biot_mimic_iii_preprocessing_2.py --name biot_mimic_iii --num_threads 1 --fs 125 --target_fs 200 --token_size 200 --window_length 10 --window_overlap 5 --ecg True --sig2sig True --plot True


# Vital DB Preprocessing
#python vital_db_preprocessing.py --name vital_db --num_threads 10 --sig2sig True --window_length 5 --window_overlap 3 > ./data_logs/vital_db.log
#python online_dataset.py --name vital_db --ecg True --sig2sig True --input_seq_len_s 5 --fs 125