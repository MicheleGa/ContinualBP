# MIMIC III Preprocessing
python preprocessing.py --name mimic_iii --num_threads 10 --window_length 5 --window_overlap 3 > ./data_logs/mimic_iii.log
python preprocessing.py --name mimic_iii --num_threads 10 --window_length 5 --window_overlap 3 --plot True
python dataset.py --name mimic_iii --fs 125 --input_seq_len_s 5 --ecg True --resp True --plot True
