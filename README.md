# Resource-Efficient Continual Learning for Continuous Blood Pressure Estimation on Edge Devices

Code repository of the paper **Resource-Efficient Continual Learning for Continuous Blood Pressure Estimation on Edge Devices**, accepted at the *The International Symposium on Edge intelligence, Trustworthy and Decentralized Artificial Intelligence (iEDGE 2026)*. 

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
├── deployment/
│   └── ...
├── figs/
│   └── ...
├── logs/
│   └── ...
├── models/
│   ├── personalizer.py
│   ├── pretrainer.py
│   ├── Proto.py
│   └── ...
├── scripts/
│   ├── deployment_pi.sh
│   ├── pretraining_proto.sh
│   └── ...
├── tensorboard/
│   └── ...
├── training_utils/
│   ├── helpers.py
│   └── metrics.py
├── .gitignore
├── LICENSE
├── deployment_personalization.py
├── drift_detector_calibration.py
├── personalization_drift_aware.py
├── personalization.py
├── pretraining.py
├── README.md 
└── results_analysis.py
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
conda create -n ps_dnn python=3.10.12
conda activate ps_dnn
conda install numpy matplotlib scikit-learn seaborn pandas markdown tensorboard
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install thop torchinfo pyCompare lmdb pyampd wfdb==4.0.0 linear_attention_transformer mat73 transformers learn2learn
conda install -c conda-forge emd-signal
```

Alternatively, it is possible to start a python virtualenv as follows (using python 3.10.12):

```bash
python -m venv venv
source ./venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.24.3 matplotlib==3.9.2 scikit-learn==1.6.1 seaborn==0.13.2 pandas==1.5.3 markdown==3.4.1 tensorboard==2.17.0
pip install thop torchinfo==1.8.0 pyCompare lmdb pyampd wfdb==4.0.0 
pip install PyWavelets==1.5.0 EMD-signal==1.6.4 einops mat73 neurokit2==0.2.12 PyYAML transformers linear_attention_transformer
```

For deployment on Raspberry Pi/Google Pixel

```bash
python -m venv venv
source ./venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install numpy==1.24.3 matplotlib==3.9.2 scikit-learn==1.6.1 seaborn==0.13.2 pandas==1.5.3 markdown==3.4.1
pip install thop torchinfo==1.8.0 pyCompare PyYAML tqdm psutil
```

## Data Provisioning & Preprocessing

### Provisioning

The code in this repository adopt the open-source dataset [Pulse DB](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full) to support the reproducibility of this work for continuous BP estimation with resource-aware CL.
The Pulse DB contatins two data sources, [MIMIC III](https://physionet.org/content/mimic3wdb-matched/1.0/) and [Vital DB](https://vitaldb.net/dataset/).
To download them, enter the *data/pulse_db* folder:

```bash
cd ./data/pulse_db
./download_and_unzip.sh
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
python pulse_db_preprocessing.py --input_folder ./pulse_db/mimic_iii_npz --index_file_name ./pulse_db/mimic_iii_index.csv --name pulse_db_mimic_iii_percentile --num_threads 8 --normalization percentile > ./data_logs/pulse_db_mimic_iii_percentile_preprocessing.log

# Test dataloaders & Plot Statistics
python dataset.py --name pulse_db_mimic_iii_percentile --fs 125 --input_seq_len_s 10 --plot --loader_worker 10 --index_file_name ./pulse_db/mimic_iii_index.csv --plot
```

the same steps must be performed for the Vital DB dataset.
Then stage-specific (pretraining/personalization) scripts can be run to test the datalaoders:

```bash
# Pretraining - Test meta-dataloaders & Plot Statistics with MIMIC III
python meta_dataloaders.py --dataset_name pulse_db_mimic_iii_percentile --k_support 4 --k_query 4 --plot --index_file_name ./pulse_db/mimic_iii_index.csv

# Personalization - Test Online Dataset & Plot Statistics with Vital DB (subject subset C)
python online_dataset.py --name pulse_db_vital_db_percentile --input_seq_len_s 10 --fs 125 --plot --personalization_batch_size 4 --num_batches 72 --num_blocks 1
```

in particular, *online_dataset.py* can also run the pareto optimization to find the values for dividing the Vital DB into the three subset 1/2/3. 

## Framework Execution

### DNN Architecture

We used the convolutional neural network (CNN) + gated recurrent unit (GRU) + fully-connected layer from [FewShotBP](https://github.com/fanfeiyi/FewShotBP). The original model is inside the *models* folder, specifically inside the *ResgruNet.py* script. We modified it to adapt it for streaming personalization by substituting the batch normalization with group/layer normalization. The resulting architecture can be found at *models/Proto.py*. We alsoe xperimetned with additional DNN architectures like the transformer-based architectecture ([BIOT](https://github.com/ycq091044/BIOT)) and the time convolutional neural network architecture ([TCN](https://github.com/locuslab/TCN)). To run a forward pass of the DNNs on dummy inputs, execute the following commands from the project repository root:


```bash
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 4 
```

that also prints the summary with of the DNN along with MACs/# params from [thop](https://github.com/ultralytics/thop).  

### Pretraining

After data provisioning/preprocessing and testing the DNN architecture, to perform pretraining, run the dedicated bash script from the scripts directory:  

```bash
./pretraining_proto.sh
```

which can run several different experiments in sequence. The bash commands to reproduce the results are:

```bash
experiment_name="proto_ppg_percentile_group_layer_norm_kq_4"
mkdir "logs/$experiment_name"
cd ./models
python Proto.py \
    --fs 125 \
    --input_seq_len_s 10 \
    --embed_dim 128 \
    --batch_size 4 \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.Proto \
    --dataset_name pulse_db_mimic_iii_percentile \
    --expname "$experiment_name" \
    --loader_worker 4 \
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
    --k_support 4 \
    --k_query 4 \
    --meta_batch_size 4 \
    --meta_lr_schedule 'cosine' \
    --meta_lr 0.001 \
    --meta_lr_scheduler_eta_min 0.00001 \
    --inner_adapt 'head' \
    --inner_opt 'adam' \
    --inner_lr_schedule 'cosine' \
    --inner_lr 0.01 \
    --inner_lr_min 0.005 \
    --inner_steps_schedule 'cosine' \
    --inner_steps 5 \
    --inner_steps_max 10 \
    --eval_lr 0.005 \
    --eval_steps 10 \
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

After pretraining, the personalization stage can be executed to reproduce the results from the paper by running the dedicated bash scripts from the scripts directory. Instead, to obtain the profiling of the personalization stage running on a [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/) and a [Pixel 10a](https://store.google.com/product/pixel_10a), it is first fundamental to follow the additional readme in the *deployment* folder. After that the *results_analys.py* script can be used to log the final results, which should be the following.

| Set | Algorithm | AE ↓ | BWT ↓ | ME ↓ (<5 mmHg) | STD ↓ (<8 mmHg) | BHS ↑ (A) |
|---|---|---:|---:|---:|---:|:---:|
| 1 | cumulative-mean | 8.0 ± 0.0 / 4.2 ± 0.0 | 0.3 ± 0.0 / 0.2 ± 0.0 | -0.4 ± 0.0 / -0.1 ± 0.0 | 11.5 ± 0.0 / 6.2 ± 0.0 | D / A |
|  | no-adapt | 13.4 ± 0.0 / 10.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 3.8 ± 0.0 / 7.2 ± 0.0 | 16.4 ± 0.0 / 10.6 ± 0.0 | D / D |
|  | first-batch | 13.2 ± 0.1 / 6.2 ± 0.1 | 0.0 ± 0.0 / 0.0 ± 0.0 | 0.9 ± 0.2 / 0.3 ± 0.1 | 17.6 ± 0.1 / 8.4 ± 0.1 | D / B |
|  | online | 9.6 ± 0.2 / 5.2 ± 0.1 | 6.0 ± 0.2 / 3.1 ± 0.1 | 0.8 ± 0.0 / 0.5 ± 0.0 | 5.9 ± 0.0 / 3.6 ± 0.0 | A / A |
|  | online* | 12.1 ± 0.3 / 6.7 ± 0.2 | 3.2 ± 0.2 / 2.2 ± 0.1 | 8.8 ± 0.1 / 3.8 ± 0.0 | 23.0 ± 0.1 / 11.5 ± 0.0 | C / B |
|  | **feat.replay** | **4.5 ± 0.1 / 2.6 ± 0.1** | **0.8 ± 0.1 / 0.5 ± 0.1** | **0.7 ± 0.0 / 0.5 ± 0.0** | **5.9 ± 0.0 / 3.5 ± 0.0** | **A / A** |
|  | LwF | 9.7 ± 0.2 / 5.3 ± 0.1 | 5.9 ± 0.2 / 3.2 ± 0.1 | 1.1 ± 0.0 / 0.6 ± 0.0 | 6.2 ± 0.0 / 3.6 ± 0.0 | A / A |
|  | EWC | 9.6 ± 0.4 / 5.2 ± 0.2 | 6.1 ± 0.4 / 3.2 ± 0.2 | 0.8 ± 0.1 / 0.5 ± 0.0 | 5.8 ± 0.0 / 3.6 ± 0.0 | A / A |
|  | AGEM | 4.9 ± 0.1 / 2.8 ± 0.1 | 1.0 ± 0.2 / 0.6 ± 0.1 | 0.8 ± 0.1 / 0.5 ± 0.0 | 6.1 ± 0.0 / 3.6 ± 0.0 | A / A |
| 2 | cumulative-mean | 10.3 ± 0.0 / 5.3 ± 0.0 | 0.3 ± 0.0 / 0.2 ± 0.0 | -1.5 ± 0.0 / -0.6 ± 0.0 | 14.1 ± 0.0 / 7.4 ± 0.0 | D / B |
|  | no-adapt | 13.3 ± 0.0 / 9.6 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 3.8 ± 0.0 / 6.3 ± 0.0 | 16.6 ± 0.0 / 10.6 ± 0.0 | D / D |
|  | first-batch | 16.8 ± 0.2 / 8.5 ± 0.1 | 0.0 ± 0.0 / 0.0 ± 0.0 | -3.5 ± 0.2 / -1.5 ± 0.1 | 22.3 ± 0.2 / 11.3 ± 0.1 | D / D |
|  | online | 12.9 ± 0.2 / 7.1 ± 0.1 | 9.2 ± 0.2 / 5.0 ± 0.1 | 0.8 ± 0.0 / 0.4 ± 0.0 | 7.7 ± 0.0 / 4.4 ± 0.0 | B / A |
|  | online* | 15.9 ± 0.2 / 8.6 ± 0.2 | 7.0 ± 0.1 / 4.1 ± 0.2 | 8.5 ± 0.1 / 3.5 ± 0.1 | 23.9 ± 0.1 / 11.6 ± 0.0 | C / B |
|  | **feat.replay** | **6.7 ± 0.1 / 3.8 ± 0.0** | **2.3 ± 0.1 / 1.4 ± 0.0** | **0.5 ± 0.0 / 0.3 ± 0.0** | **7.9 ± 0.0 / 4.3 ± 0.0** | **B / A** |
|  | LwF | 12.7 ± 0.2 / 7.0 ± 0.1 | 8.6 ± 0.3 / 4.8 ± 0.1 | 1.1 ± 0.0 / 0.5 ± 0.0 | 7.9 ± 0.0 / 4.4 ± 0.0 | B / A |
|  | EWC | 12.7 ± 0.2 / 7.0 ± 0.1 | 9.1 ± 0.2 / 4.9 ± 0.1 | 0.7 ± 0.0 / 0.4 ± 0.0 | 7.7 ± 0.0 / 4.4 ± 0.0 | B / A |
|  | AGEM | 7.1 ± 0.1 / 3.9 ± 0.1 | 2.5 ± 0.1 / 1.3 ± 0.1 | 0.6 ± 0.0 / 0.4 ± 0.0 | 8.0 ± 0.0 / 4.4 ± 0.0 | B / A |
| 3 | cumulative-mean | 11.4 ± 0.0 / 5.9 ± 0.0 | 0.3 ± 0.0 / 0.2 ± 0.0 | -1.4 ± 0.0 / -0.2 ± 0.0 | 15.6 ± 0.0 / 8.3 ± 0.0 | D / B |
|  | no-adapt | 12.9 ± 0.0 / 10.0 ± 0.0 | 0.0 ± 0.0 / 0.0 ± 0.0 | 3.1 ± 0.0 / 5.8 ± 0.0 | 16.2 ± 0.0 / 11.3 ± 0.0 | D / D |
|  | first-batch | 20.6 ± 0.2 / 10.7 ± 0.1 | 0.0 ± 0.0 / 0.0 ± 0.0 | -5.5 ± 0.0 / -2.3 ± 0.2 | 26.8 ± 0.2 / 14.2 ± 0.1 | D / D |
|  | online | 16.5 ± 0.4 / 8.7 ± 0.2 | 12.8 ± 0.4 / 6.5 ± 0.2 | 0.8 ± 0.0 / 0.4 ± 0.0 | 10.2 ± 0.0 / 5.8 ± 0.0 | C / A |
|  | online* | 19.1 ± 0.2 / 10.6 ± 0.3 | 10.3 ± 0.2 / 6.0 ± 0.3 | 8.4 ± 0.1 / 3.4 ± 0.1 | 24.4 ± 0.1 / 12.1 ± 0.0 | D / B |
|  | **feat.replay** | **7.8 ± 0.2 / 4.5 ± 0.1** | **2.8 ± 0.2 / 1.8 ± 0.1** | **0.3 ± 0.0 / 0.3 ± 0.0** | **9.6 ± 0.0 / 5.3 ± 0.0** | **C / A** |
|  | LwF | 15.4 ± 0.3 / 8.4 ± 0.2 | 11.2 ± 0.3 / 6.2 ± 0.2 | 1.1 ± 0.0 / 0.5 ± 0.0 | 10.1 ± 0.0 / 5.6 ± 0.0 | C / A |
|  | EWC | 16.1 ± 0.3 / 8.5 ± 0.2 | 12.4 ± 0.3 / 6.4 ± 0.2 | 0.8 ± 0.0 / 0.4 ± 0.0 | 10.2 ± 0.0 / 5.7 ± 0.0 | C / A |
|  | AGEM | 8.1 ± 0.2 / 4.7 ± 0.1 | 3.2 ± 0.2 / 2.0 ± 0.1 | 0.5 ± 0.0 / 0.4 ± 0.0 | 9.8 ± 0.0 / 5.4 ± 0.0 | C / A |

**Notes:** Personalization results (over three seeds) on VitalDB for continuous SBP/DBP estimation from PPG, divided into the three subject sets 1/2/3. Parentheses indicate the target thresholds for the clinical standards for both SBP/DBP. CL algorithms are in light gray.

| Metric | Always | MMD | Random | LSDD |
|---|---:|---:|---:|---:|
| AE ↓ | 4.5 ± 0.1 | **4.9 ± 0.2** | 4.9 ± 0.1 | 7.9 ± 0.2 |
| BWT ↓ | 0.8 ± 0.1 | **1.0 ± 0.1** | 0.9 ± 0.1 | -0.1 ± 0.1 |
| ME ↓ | 0.7 ± 0.0 | **0.8 ± 0.1** | 0.8 ± 0.1 | 1.1 ± 0.3 |
| STD ↓ | 5.9 ± 0.0 | **6.1 ± 0.0** | 6.4 ± 0.1 | 11.5 ± 0.3 |
| BHS ↑ | A | **A** | A | D |
| Skip% ↑ | 0.0 ± 0.0 | **26.8 ± 12.8** | 26.8 ± 12.8 | 84.2 ± 9.5 |

**Notes:** Drift-aware personalization results (three seeds) for continuous SBP estimation from PPG (DBP omitted for conciseness). We report the fraction of skipped updates relative to the always-on baseline.

| Emb. | ME ↓ | STD ↓ | BHS ↑ | Skip% ↑ |
|---|---:|---:|:---:|---:|
| 128 | 0.8 / 0.8 / 0.7 | 6.1 / 6.1 / 6.4 | A / A / A | 26.8 / 31.0 / 42.7 |
| 32 | 1.4 / 1.4 / 1.4 | 6.3 / 6.4 / 6.7 | A / A / A | 33.4 / 36.7 / 47.7 |
| **16** | 1.9 / 1.8 / **1.9** | 6.6 / 6.7 / **7.1** | A / A / **A** | 36.7 / 39.6 / **49.9** |

**Notes:** Ablation study on MMD, over smaller embedding (128/32/16) and buffer (64/32/16) sizes. For conciseness, we report the SBP results averaged over three seeds but w/o variance. For embedding size 8, MMD was numerically unstable.

| Metric | Raspberry Pi 5 | Google Pixel 10a |
| :--- | :---: | :---: |
| **_Estimated Communication Requirements_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Setting A | 14.2 kB | 14.2 kB |
| &nbsp;&nbsp;&nbsp;&nbsp;Setting B | 1.4 MB | 1.4 MB |
| **_Profiled Comput. Requirements $\sim$ Update Latency Breakdown (ms)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;feat. extraction | 21.8 ± 0.6 | 65.0 ± 14.8 |
| &nbsp;&nbsp;&nbsp;&nbsp;BP prediction | 0.6 ± 0.0 | 1.0 ± 0.2 |
| &nbsp;&nbsp;&nbsp;&nbsp;drift detection | 4.4 ± 0.1 | 15.0 ± 3.0 |
| &nbsp;&nbsp;&nbsp;&nbsp;head adapt. | 13.8 ± 3.5 | 14.8 ± 4.8 |
| &nbsp;&nbsp;&nbsp;&nbsp;detector reinit. | 204.1 ± 56.1 | 82.4 ± 34.3 |
| **_Cumulative Latency for 88 subj. (min)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;MMD | 33.3 ± 0.9 | 29.7 ± 2.3 |
| &nbsp;&nbsp;&nbsp;&nbsp;Always-on | 12.7 ± 0.0 | 21.8 ± 1.4 |
| **_Profiled Memory Requirements $\sim$ Avg. Memory (MB)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp; | 371.9 ± 0.1 | 355.9 ± 0.2 |
| **_Profiled Energy Requirements $\sim$ Avg. Energy (J)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Update | 0.1 ± 0.0 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;Subject | 8.5 ± 1.2 | - |
| **_Profiled Avg. Update Frequency Reduction (%)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp; | 50.3 ± 12.5 | 50.2 ± 13.0 |
| **_Profiled Avg. Annotations required for one subj. (# samples)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;MMD | 143.0 ± 31.9 | 143.3 ± 32.9 |
| &nbsp;&nbsp;&nbsp;&nbsp;Always-on | 288.0 ± 0.0 | 288.0 ± 0.0 |

**Notes:** Resource profiling (over three seeds) of feature replay with MMD, embedding size 16 and buffer size 16.

| Metric | Raspberry Pi 5 | Google Pixel 10a |
| :--- | :---: | :---: |
| **_Estimated Communication Requirements_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Setting A | 110.6 kB | 110.6 kB |
| &nbsp;&nbsp;&nbsp;&nbsp;Setting B | 2.9 MB | 2.9 MB |
| **_Profiled Comput. Requirements ~ Update Latency Breakdown (ms)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;feat. extraction | 59.5 ± 0.7 | 696.8 ± 17.1 |
| &nbsp;&nbsp;&nbsp;&nbsp;BP prediction | 0.9 ± 0.0 | 3.8 ± 0.1 |
| &nbsp;&nbsp;&nbsp;&nbsp;drift detection | 4.5 ± 0.1 | 22.4 ± 0.5 |
| &nbsp;&nbsp;&nbsp;&nbsp;head adapt. | 22.8 ± 5.8 | 60.1 ± 15.9 |
| &nbsp;&nbsp;&nbsp;&nbsp;detector reinit. | 248.6 ± 66.0 | 93.8 ± 26.1 |
| **_Cumulative Latency for 88 subj. (min)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;MMD | 43.0 ± 0.0 | 102.7 ± 0.0 |
| &nbsp;&nbsp;&nbsp;&nbsp;Always-on | 18.8 ± 0.0 | 96.8 ± 0.0 |
| **_Profiled Memory Requirements ~ Avg. Memory (MB)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp; | 375.7 ± 0.2 | 360.5 ± 0.2 |
| **_Profiled Energy Requirements ~ Avg. Energy (J)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Update | 0.2 ± 0.0 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;Subject | 13.0 ± 1.6 | - |
| **_Profiled Avg. Update Frequency Reduction (%)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp; | 41.1 ± 14.8 | 42.6 ± 14.8 |
| **_Profiled Avg. Annotations required for one subj. (# samples)_** | | |
| &nbsp;&nbsp;&nbsp;&nbsp;MMD | 169.5 ± 42.5 | 165.2 ± 42.6 |
| &nbsp;&nbsp;&nbsp;&nbsp;Always-on | 288.0 ± 0.0 | 288.0 ± 0.0 |

**Notes:** Resource profiling of feature replay with MMD, embedding size 32 and buffer size 16.