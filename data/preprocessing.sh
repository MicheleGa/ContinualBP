# MIMIC III Preprocessing
python mimic_iii_preprocessing.py --name mimic_iii --num_threads 10 --ecg True --sig2sig True --window_length 5 --window_overlap 3 --ema_std False > ./data_logs/mimic_iii.log
python mimic_iii_preprocessing.py --name mimic_iii --num_threads 1 --ecg True --sig2sig True --window_length 5 --window_overlap 3 --ema_std False --plot True
python dataset.py --name mimic_iii --fs 125 --sig2sig True --input_seq_len_s 5 --ecg True --plot True
python dataset.py --name mimic_iii --fs 125 --sig2sig True --input_seq_len_s 5 --ecg True --mix_pretraining_subject_samples False --plot True
python dataset_ssl.py --name mimic_iii --fs 125 --sig2sig True --input_seq_len_s 5 --ecg True --mix_pretraining_subject_samples False --plot True



#python mimic_iii_xl_preprocessing.py --name mimic_iii_xl --num_threads 10 --sig2sig True --window_length 5 --window_overlap 2.5 --ecg True > ./data_logs/mimic_iii_xl.log
#python dataset.py --name mimic_iii_xl --fs 125 --sig2sig True --input_seq_len_s 5 --ecg True --resp True --plot True


# Vital DB Preprocessing
#python vital_db_preprocessing.py --name vital_db --num_threads 10 --sig2sig True --window_length 5 --window_overlap 3 > ./data_logs/vital_db.log
#python online_dataset.py --name vital_db --ecg True --sig2sig True --input_seq_len_s 5 --fs 125