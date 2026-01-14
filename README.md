# Resource-Aware Continual Learning for Continuous Blood Pressure Estimation on Edge Devices

This is a PyTorch implementation of the Continual Learning (CL) experiments with deep neural networks (DNNs) for continuous Blood Pressure (BP) under resource constraints typical of edge devices. 

Readme Overview:

1) Environment Setup
2) Data Provisioning & Preprocessing
3) Framework Execution

Repository filetree:

```
ContinualBP/
├── checkpoints/
├── data/
│   ├── data_figs/
│   │   └── ...
│   ├── data_logs/
│   │   └── ...
│   ├── lmdb/
│   │   └── ...
│   ├── preprocessing_utils/
│   │   └── ...
│   ├── pulse_db/
│   │   ├── mimic_iii_npz 
│   │   │   └── ...
│   │   ├── vital_db_npz
│   │   │   └── ...
│   │   └── ...
│   ├── dataset.py
│   ├── meta_dataloaders.py
│   ├── online_dataset.py
│   ├── preprocessing.py
│   └── pulse_db_preprocessing.sh
├── figs/
│   └── ...
├── logs/
│   └── ...
├── models/
│   ├── personalizer.py
│   ├── pretrainer.py
│   ├── Proto.py
│   └── ...
├── tensorboard/
│   └── ...
├── training_utils/
│   ├── helpers.py
│   ├── metrics.py
│   └── ...
├── .gitignore
├── LICENSE
├── personalization.py
├── personalization_biot.sh
├── personalization_drift_aware.sh
├── personalization_normalization.sh
├── personalization_proto.sh
├── personalization_resgrunet.sh
├── personalization_scenarios.sh
├── personalization_split_blocks.sh
├── personalization_tcn.sh
├── pretraining.py
├── pretraining_biot.sh
├── pretraining_proto.sh
├── pretraining_resgrunet.sh
├── pretraining_tcn.sh
├── README.md 
├── results_analysis.py
└── results_analysis.sh
```

## Environment Setup

Environment Setup section overview:

> - CUDA Toolkit Installation
> - Packages & Libraries Setup

If your machine already has CUDA and Nvidia Drivers you may skip this section and look at the Conda setup for package management.

### CUDA Toolkit Installation

#### Preliminary steps

Ubuntu 22.04 (Jammy Jellyfish) kernel setup:

```bash
 sudo apt install --reinstall linux-image-generic
 sudo apt install --reinstall linux-headers-generic
 ```

Be sure of having GCC compiler version 12.3.0

```bash
gcc --version
```

Purge every kind of nvidia/cuda related file

```bash
sudo apt remove --purge '^nvidia-.*'
sudo apt remove --purge '^libnvidia-.*'
sudo rm /etc/X11/xorg.conf | true
sudo rm /etc/X11/xorg.conf.d/90-nvidia-primary.conf | true
sudo rm /usr/share/X11/xorg.conf.d/10-nvidia.conf | true
sudo rm /usr/share/X11/xorg.conf.d/11-nvidia-prime.conf | true
sudo rm /etc/modprobe.d/nvidia-kms.conf | true
sudo rm /lib/modprobe.d/nvidia-kms.conf | true
sudo apt update -y && sudo apt full-upgrade -y && sudo apt autoremove -y && sudo apt clean -y && sudo apt autoclean -y
```

#### Nvidia Drivers

Open the Software & Update app, and install the nvidia-drivers-550 (propertary, tested). Then reboot and test the installation with:

```bash
nvidia-smi
```

#### CUDA Toolkit

Install the CUDA Toolkit 12.4.0:

```bash
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

Add the following lines to the bashrc file (which require a terminal restart)

```bash
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Test the installation by cloning the following repo:

```bash
cd ~/Downloads
git clone git@github.com:NVIDIA/cuda-samples.git
```

and then:

```bash
cd cuda-samples/Samples/1_Utilities/deviceQuery
make
./deviceQuery
cd ../bandwidthTest
make
./bandwidthTest
```

Observe the output and compare them to the one reported in the official [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-the-installation).

#### CuDNN Installation

From the official [guide](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/install-guide/index.html):

```bash
sudo apt-get install zlib1g
```

and then 

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.7.0/local_installers/cudnn-local-repo-ubuntu2204-9.7.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.7.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.7.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

and test on MNIST:

```bash
sudo apt-get -y install libcudnn9-samples
cd /usr/src/cudnn_samples_v9
sudo make clean && sudo make
./mnistCUDNN
```

You should get a _Test Passed!_

### Packages & Libraries Setup

You can install conda following the official instructions in their [website](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

Then, from the base environment:

```bash
conda create -n ps_dnn python=3.9.0
conda activate ps_dnn
conda install numpy matplotlib scikit-learn seaborn pandas markdown tensorboard
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install thop torchinfo pyCompare lmdb pyampd wfdb==4.0.0 linear_attention_transformer mat73 transformers learn2learn
conda install -c conda-forge emd-signal
conda install -c conda-forge pywavelets
```

Alternatively, it is possible to start a python virtualenv as follows (using python 3.10.12):

```bash
python -m venv venv
source ./venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.24.3 matplotlib==3.9.2 scikit-learn==1.6.1 seaborn==0.13.2 pandas==1.5.3 markdown==3.4.1 tensorboard==2.17.0
pip install netron==8.1.5 thop torchinfo==1.8.0 pyCompare lmdb pyampd wfdb==4.0.0 
pip install PyWavelets==1.5.0 EMD-signal==1.6.4 einops linear_attention_transformer mat73 transformers learn2learn
```

## Data Provisioning & Preprocessing

### Provisioning

The code in this repository adopt the open-source dataset [Pulse DB](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full) to support the reproducibility of this work for continuous BP estimation with resource-aware CL.
The Pulse DB contatins two data sources, [MIMIC III](https://physionet.org/content/mimic3wdb-matched/1.0/) and [Vital DB](https://vitaldb.net/dataset/).
To download them, enter the *data/pulse_db* folder:

```bash
cd ./data/pulse_db
./downaload_and_unzip.sh
```

The downlaod process may take a while and can also fail as reported in the Pulse DB, github [repo](https://github.com/pulselabteam/PulseDB).
We invite to follow their guidelines to interact with Pulse DB. 

### Preprocessing

After successfull download and unzip two new directories must be present: *data/pulse_db/PulseDB_MIMMIC* and *data/pulse_db/PulseDB_Vital*. Run

```bash
./convert2numpy.sh
```
to convert the downlaoded files of MIMIC III/Vital DB from matlab to numpy arrays. The script also collects the demographics information available in the two datasets into two files (CSVs) indexed on the subject id. Two new directories are created in *data/pulse_db*, namely *data/pulse_db/mimic_iii_npz* and *data/pulse_db/vital_db_npz*. If the scripts run without errors until here, *data/pulse_db/PulseDB_MIMMIC* and *data/pulse_db/PulseDB_Vital* can be safely deleted to save storage.

After unzipping the dataset, the preprocessing pipeline can be applied by running the script in the *data* directory, so, right after the unzip/conversion commands (assuming the *ps_dnn* or *venv* env is active):

```bash
cd ..
./preprocessing.sh
```

Importantly, *preprocessing.sh* can launch different kind of preprocessing in sequence, each fo which can be adapted to the specific use case with the reposiory of signal processing functions in *preprocessing_utils*. For example, to preprocess MIMIC III:

```bash
# Plot Sample Signals
python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_percentile --num_threads 1 --plot --normalization percentile

# Build lmdb & Percentile normalize
python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_percentile --num_threads 8 --normalization percentile > ./data_logs/pulse_db_mimic_iii_z_score_preprocessing.log

# Test dataloaders & Plot Statistics
python dataset.py --name pulse_db_mimic_iii_percentile --fs 125 --input_seq_len_s 10 --plot --loader_worker 10 --index_file_name ./pulse_db/mimic_iii_index.csv --plot
```

the same steps must be performed for the Vital DB dataset.
Then stage-specific (pretraining/personalization) scripts can be run to test the datalaoders:

```bash
# Pretraining - Test meta-dataloaders & Plot Statistics with MIMIC III
python meta_dataloaders.py --dataset_name pulse_db_mimic_iii_percentile --k_support 16 --k_query 16 --plot --index_file_name ./pulse_db/mimic_iii_index.csv

# Personalization - Test Online Dataset & Plot Statistics with Vital DB (subject subset C)
python online_dataset.py --name pulse_db_vital_db_percentile --input_seq_len_s 10 --fs 125 --plot --personalization_batch_size 16 --validation_batch_size 16 --valid_runs_number 3 --num_train_val 3 --split_blocks 1
```

in particular, *online_dataset.py* also run the pareto optimization to find the values for dividing the Vital DB into the three subset A/B/C. 

## Framework Execution

### DNN Architecture

We used the convolutional neural netowrk (CNN) + gated recurrent unit (GRU) + fully-connected layer from [FewShotBP](https://github.com/fanfeiyi/FewShotBP). The original model is inside the *models* folder, specifically inside the *ResgruNet.py* script. We modified it to adapt it for streaming personalization by substituting the batch normalization with group/layer normalization. The resulting architecture can be found at *models/Proto.py*. We alsoe xperimetned with additional DNN architectures like the transformer-based architectecture ([BIOT](https://github.com/ycq091044/BIOT)) and the time convolutional neural network architecture ([TCN](https://github.com/locuslab/TCN)). To run a forward pass of the DNNs on dummy inputs, execute the following commands from the project repository root:


```bash
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 1 
```

that also prints the summary with of the DNN along with MACs/# params from [thop](https://github.com/ultralytics/thop).  

### Pretraining

After data provisioning/preprocessing and testing the DNN architecture, to perform pretraining, run the dedicated bash script from the project repository root:  

```bash
./pretraining_proto.sh
```

which can run several different experiments in sequence. The bash commands to reproduce the results are:

```bash
experiment_name="proto_ppg_percentile_group_layer_norm"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 16 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.Proto \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 128 \
    --stage1_pre_train_lr 0.001 \
    --stage1_pre_train_scheduler_eta_min 0.00001 \
    --weight_decay 0.0001 \
    --stage1_epochs 10 \
    --criterion "SmoothL1Loss" \
    --max_meta_epochs 200 \
    --k_support 16 \
    --k_query 16 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'constant' \
    --inner_lr 0.01 \
    --inner_steps_schedule 'constant' \
    --inner_steps 8 \
    --eval_lr 0.01 \
    --eval_steps 8 \
    > "./logs/$experiment_name/${experiment_name}_training.log"
```

where the frst python script run the the model on a dummy input to register the memory footprint and number of operations (output in *logs/$experiment_name*), while the second starts the actual pretraining. It is possible to follow the training progression by openinig a second terminal and run tensorboard with the following command:

```bash
tensorboard --logdir tensorboard/$experiment_name
```

while it is also possible to look into training logs in the corresponding folder (again, *logs/$experiment_name*).
Since pretraining may take a while the following command can be useful to launch the scripts in background:

```bash
nohup ./your_script_runner.sh > /location/of/the/output/file.log 2>&1 &
```

### Personalization

After pretraining, the personalization stage can be executed by running the dedicated bash script from the project repository root:  

```bash
./personalization_proto.sh
```

which can also run several different experiments in sequence. The bash commands to reproduce the results are:

```bash
experiment_name="proto_ppg_percentile_group_layer_norm"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 1 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python personalization.py \
    --model models.Proto \
    --dataset_name pulse_db_vital_db_percentile \
    --expname "$experiment_name" \
    --loader_worker 10 \
    --gpu 0 \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --pretrained_model_ckpt_path ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm_best_maml \
    --pretraining_feats_stats ./checkpoints/proto_ppg_percentile_group_layer_norm/proto_ppg_percentile_group_layer_norm-Proto-2026_01_06-11_03_38/proto_ppg_percentile_group_layer_norm_embedding_stats.npz \
    --criterion 'SmoothL1Loss' \
    --personalization_lr 0.01 \
    --personalization_steps 8 \
    --personalization_batch_size 16 \
    --validation_batch_size 16 \
    --valid_runs_number 3 \
    --num_train_val 3 \
    --split_blocks 1 \
    --replay_buffer_size 64 \
    --replay_batch_size 16 \
    --inner_adapt 'head' \
    --plot_personalization \
    --setup_type 'fixed' \
    > "./logs/$experiment_name/${experiment_name}_personalization_training.log"
```

where the frst python script run the the model on a dummy input to register the memory footprint and number of operations (output in *logs/$experiment_name*), while the second starts the actual personalization on the subject subset C of the Vital DB. It is possible to reproduce the results for subset A/B by running

```bash
./personalization_scenarios.sh
```

The final results should be the following:

| Subset | Algorithm      | AA (SBP/DBP)↓ | BWT (SBP/DBP)↓  | MAE (SBP/DBP)↓ | ME ± STD (SBP/DBP)↓       | BHS (SBP/DBP)↑ |
| ------ | -------------- | ------------- | --------------  | -------------  | ------------------------ | ------------- |
| A      | no adapt       | 13.17 / 10.26 | 0.00 / 0.00     | 13.23 / 10.25  | 2.87±16.38 / 7.49±10.62  | D / D         |
|        | first batch    | 10.92 / 5.49  | 0.00 / 0.00     | 11.18 / 5.82   | 0.54±14.89 / 0.96±7.88   | D / B         |
|        | online         | 8.75 / 4.77   | 4.30 / 2.24     | 6.33 / 3.72    | 0.40±9.76 / 0.65±6.01    | B / A         |
|        | online*        | 10.84 / 5.88  | −13.62 / −4.38  | 29.48 / 13.38  | 26.00±41.14 / 9.21±21.17 | D / D         |
|        | **feature replay** | **5.75 / 3.29**   | **1.09 / 0.62**     | **6.02 / 3.56**    | **0.68±9.08 / 0.79±5.65**    | **B / A**        |
|        | LwF            | 8.54 / 4.70   | 4.17 / 2.13     | 6.24 / 3.70    | 0.59±9.55 / 0.63±5.95    | B / A         |
|        | EWC            | 8.54 / 4.78   | 4.11 / 2.27     | 6.30 / 3.68    | 0.45±9.68 / 0.63±5.94    | B / A         |
|        | AGEM           | 6.27 / 3.50   | 1.45 / 0.78     | 6.21 / 3.62    | 0.54±9.32 / 0.70±5.77    | B / A         |

| Subset | Algorithm      | AA (SBP/DBP)↓ | BWT (SBP/DBP)↓  | MAE (SBP/DBP)↓ | ME ± STD (SBP/DBP)↓       | BHS (SBP/DBP)↑ |
| ------ | -------------- | ------------- | --------------  | -------------  | ------------------------ | ------------- |
| B      | no adapt       | 14.38 / 10.17 | 0.00 / 0.00     | 14.47 / 10.26  | 3.27±18.28 / 6.76±11.17  | D / D         |
|        | first batch    | 14.39 / 7.62  | 0.00 / 0.00     | 14.65 / 7.90   | −0.54±19.12 / 0.27±10.26 | D / C         |
|        | online         | 11.34 / 6.34  | 6.56 / 3.69     | 8.08 / 4.48    | 0.31±11.86 / 0.19±6.59   | D / A         |
|        | online*        | 14.34 / 7.85  | −6.13 / −0.82   | 26.93 / 12.36  | 20.99±40.77 / 6.70±20.12 | D / D         |
|        | **feature replay** | **8.25 / 4.63**   | **3.11 / 1.85**     | **7.73 / 4.29**    | **0.33±11.20 / 0.34±6.29**   | **C / A**         |
|        | LwF            | 11.18 / 6.14  | 6.41 / 3.48     | 7.99 / 4.52    | 0.40±11.67 / 0.32±6.62   | C / A         |
|        | EWC            | 11.27 / 6.31  | 6.41 / 3.65     | 8.11 / 4.49    | 0.41±11.94 / 0.30±6.57   | D / A         |
|        | AGEM           | 8.77 / 4.71   | 3.45 / 1.77     | 7.92 / 4.40    | 0.41±11.39 / 0.36±6.37   | C / A         |

| Subset | Algorithm      | AA (SBP/DBP)↓ | BWT (SBP/DBP)↓  | MAE (SBP/DBP)↓ | ME ± STD (SBP/DBP)↓       | BHS (SBP/DBP)↑ |
| ------ | -------------- | ------------- | --------------  | -------------  | ------------------------ | ------------- |
| C      | no adapt       | 14.81 / 10.58 | 0.00 / 0.00     | 14.82 / 10.61  | 3.53±18.45 / 6.79±11.54  | D / D         |
|        | first batch    | 12.61 / 6.36  | 0.00 / 0.00     | 12.99 / 6.73   | 0.74±17.17 / 1.22±9.02   | D / C         |
|        | online         | 10.12 / 5.35  | 5.44 / 2.74     | 7.35 / 4.13    | 0.46±11.09 / 0.53±6.42   | C / A         |
|        | online*        | 12.33 / 6.53  | −12.02 / −3.39  | 30.19 / 13.39  | 26.15±42.02 / 8.79±21.28 | D / D         |
|        | **feature replay** | **7.20 / 3.80**   | **2.15 / 1.03**     | **7.13 / 3.97**    | **0.74±10.69 / 0.69±6.16**   | **C / A**         |
|        | LwF            | 10.19 / 5.42  | 5.61 / 2.86     | 7.24 / 4.09    | 0.52±10.86 / 0.59±6.37   | C / A         |
|        | EWC            | 10.12 / 5.36  | 5.41 / 2.76     | 7.35 / 4.09    | 0.36±11.05 / 0.56±6.38   | C / A         |
|        | AGEM           | 7.47 / 3.94   | 2.29 / 1.06     | 7.23 / 4.07    | 0.67±10.80 / 0.71±6.23   | C / A         |

Instead, to obtain the resource estimation of the personalization stage and obtain more fine-grained analysis of the CL algorithm perofrmance, first the personalization stage must run with drift detection

```bash
./personalization_drift_aware.sh
```

then run for the project root folder

```bash
./results_analysis.sh
```

which will print the MACs savings and the Average Accuracy (AA) degradation. The *results_analysis.sh* bash script requires the personalization stage to first run **without** drfit detection and then **with** drift detection so that it is possible to pass to **results_analysis.py** the right folder paths (see *results_analysis.sh*).
