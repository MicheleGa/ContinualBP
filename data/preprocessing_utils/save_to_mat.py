import os
from shutil import rmtree
import argparse
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt


def save_numpy_array_to_mat(array, filename, variable_name='data'):
    """
    Saves a NumPy array to a .mat file for use in MATLAB.

    Args:
        array (numpy.ndarray): The NumPy array to save.
        filename (str): The desired filename (including .mat extension).
        variable_name (str): The variable name to use in MATLAB. Defaults to 'data'.
    """
    try:
        savemat(filename, {variable_name: array})
        print(f"Array successfully saved to {filename} with variable name '{variable_name}'.")
    except Exception as e:
        print(f"Error saving array: {e}")
        

def save_dataset(args):
    
    subjects_labels_path = [
        f.path for f in os.scandir(os.path.join(args.input_folder, 'labels'))
        if f.is_file() and f.name.endswith('.npy')
    ]
    
    subjects_ids = []
    for file in subjects_labels_path:
        subjects_ids.append(file.split('/')[-1].split('_')[0])
    
    # Always start with a clean folder
    if os.path.exists(args.output_folder):
        rmtree(args.output_folder)
    os.makedirs(args.output_folder)

    for subject_id in subjects_ids:
        
        subject_ppgs = np.load(os.path.join(args.input_folder, 'ppg', f'{subject_id}_ppg.npy'))
        subject_abps = np.load(os.path.join(args.input_folder, 'abp', f'{subject_id}_abp.npy'))
        subject_ecgs = np.load(os.path.join(args.input_folder, 'ecg', f'{subject_id}_ecg.npy'))
        
        data = np.concatenate([subject_ppgs[np.newaxis, :, :], subject_abps[np.newaxis, :, :], subject_ecgs[np.newaxis, :, :]])
        save_numpy_array_to_mat(data, os.path.join(args.output_folder, f'{subject_id}.mat'))
        

def parseargs():
    parser = argparse.ArgumentParser(description="MIMIC III Preprocessing Pipeline")
    parser.add_argument('--input_folder', default='./raw_mimic_iii', type=str, help='path to raw dataset')
    parser.add_argument('--output_folder', default='./cases', type=str, help='path to cleaned dataset')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()
    
    print("Parsed Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print()
    
    save_dataset(args)
