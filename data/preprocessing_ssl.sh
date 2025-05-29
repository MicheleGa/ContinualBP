# MIMIC III Preprocessing - Self-Supervision
# Note the 2.5 as we are trying to calculate next/prev windows of analysis, therefore enforcing less overlap
#python preprocessing_ssl.py --name mimic_iii_ssl --num_threads 10 --sig2sig True --window_length 5 --window_overlap 2.5 > ./data_logs/mimic_iii_ssl.log
python dataset_ssl.py --name mimic_iii_ssl --fs 125 --input_seq_len_s 5 --sig2sig True --input_seq_len_s 5 --ecg True --batch_size 128 --plot True