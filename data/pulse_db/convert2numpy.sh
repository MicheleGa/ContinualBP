#Pulse DB - conversion to numpy arrays

## !!! First run download_and_unzip.sh !!!

## Convert to numpy arrays
python convert2numpy.py --in_folder './PulseDB_MIMIC' --out_folder './mimic_iii_npz' --index_file 'mimic_iii_index.csv' --n_jobs 8
python convert2numpy.py --in_folder './PulseDB_Vital' --out_folder './vital_db_npz' --index_file 'vital_db_index.csv' --n_jobs 8