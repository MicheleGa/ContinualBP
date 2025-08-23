# Blood Pressure Estimation with Neural Networks

Readme Overview:

1) Environment Setup
2) Data Provisioning and Preprocessing
3) Model Development 
4) Example Usage

Repository overview:

```
ContinualBP/
├── checkpoints/
├── data/
│   ├── data_figs/
│   ├── data_logs/
│   ├── lmdb/
│   ├── preprocessing_utils/
│   │   ├── augmentations.py
│   │   └── ...
│   ├── raw_mimic_iii/
│   │   └── datavers_files.zip
│   ├── dataset.py
│   ├── preprocessing.py
│   └── preprocessing.sh
├── figs/
├── logs/
├── models/
│   ├── ResGRUNet.py
│   └── ...
├── notebooks/
│   └── bp_estimation_with_nn.ipynb
├── tensorboard/
├── training_utils/
│   ├── helpers.py
│   └── ...
├── .gitignore
├── pretraining.py
├── pretraining.sh
└── README.md
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
pip install netron thop torchinfo pyCompare lmdb pyampd wfdb==4.0.0 linear_attention_transformer
MAX_JOBS=4 pip install flash-attn --no-build-isolation
conda install -c conda-forge emd-signal
conda install -c conda-forge pywavelets
conda install lightning -c conda-forge
```

The _flas-attn_ package may be skipped as installation takes long and it is not yet employed in the models.
Alternatively, it is possible to start a python virtualenv as follows (using python 3.10.12):

```bash
python -m venv venv
source ./venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.24.3 matplotlib==3.9.2 scikit-learn==1.6.1 seaborn==0.13.2 pandas==1.5.3 markdown==3.4.1 tensorboard==2.17.0
pip install netron==8.1.5 thop torchinfo==1.8.0 pyCompare lmdb pyampd wfdb==4.0.0 
pip install PyWavelets==1.5.0 EMD-signal==1.6.4 lightning einops linear_attention_transformer mat73
```

## Data Provisioning & Preprocessing

### Provisioning ~ MIMIC III Curated Dataset

The code in this repository adopt the dataset [MIMIC III](https://physionet.org/content/mimic3wdb-matched/1.0/) that has been widely employed for BP estimation.
Recently, it has been thorougly preprocessed in the following [paper](https://www.nature.com/articles/s41597-024-04041-1), that open sourced the data provisioning step. Therefore we downloaded their dataset and in *data/raw_mimic_iii* directory and then:

```bash
cd ./data/raw_mimic_iii
unzip dataverse_files.zip
unzip abp.zip -d abp
unzip labels.zip -d labels
unzip ecg.zip -d ecg
unzip ppg.zip -d ppg
unzip resp.zip -d resp
```

### Provisioning ~ MIMIC III from PulseDB

The code in this repository explain how to downlaod the already preprocessed MIMIC III from the [PulseDB](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full) publication: [GitHub](https://github.com/pulselabteam/PulseDB/tree/main).
Additionally, after unzipping, the _.mat_ files directory have been renamed to *mimic_iii*.

### Preprocessing

After unzipping the dataset, the preprocessing pipeline can be applied by running the script in the *data* directory, so, right after the unzip commands (assuming the *ps_dnn* or *venv* env is active):

```bash
cd ..
./preprocessing.sh
```

Importantly, *preprocessing.sh* can launch different kind of preprocessing in sequence, each fo which can be adapted to the specific use case with the reposiory of signal processing functions in *preprocessing_utils*. Please note that the bash script is supposed to run the following commands for a given preprocessing pipeline:

```bash
python preprocessing.py --name mimic_iii --num_threads 10 --window_length 5 --window_overlap 3 > ./data_logs/mimic_iii.log

python preprocessing.py --name mimic_iii --num_threads 10 --window_length 5 --window_overlap 3 --plot True

python dataset.py --name mimic_iii --fs 125 --input_seq_len_s 5 --ecg True --resp True --plot True
```

that preprocess, plot the preprocessing steps, and plot the dataset statistics.

## Model Development

### Pretraining

After preprocessing, to perform pretraining, run the dedicated bash script:  

```bash
cd ..
./pretraining.sh
```

which can run several different experiments in sequence. The bash commands related to a single experiments should be based on the following template:

```bash
experiment_name="vanilla_resnet"
mkdir "logs/$experiment_name"
cd ./models
python ResGRUNet.py \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    > "../logs/$experiment_name/${experiment_name}_summary.log"
cd ..
python pretraining.py \
    --model models.ResGRUNet \
    --dataset_name mimic_iii \
    --expname "$experiment_name" \
    --loader_worker 4 \
    --ecg True \
    --fs 125 \
    --input_seq_len_s 5 \
    --proj_head_dim 256 \
    --return_embedding True \
    --set_tunable_params all \
    --mix_pretraining_subject_samples True \
    --max_training_epochs 100 \
    --lr_scheduler_enable True \
    --optimizer_type "AdamW" \
    --lr_scheduler_type "CosineAnnealingWarmupScheduler" \
    --lr 0.01 \
    --lr_scheduler_min_lr 0.00001 \
    --lr_scheduler_warmup 10 \
    --eval_every_n_epochs 1 \
    --lambda_supervised 1 \
    --lambda_ortho 0 \
    --lambda_contrastive 0 \
    --temperature 0 \
    --aug False \
    > "./logs/$experiment_name/${experiment_name}_training.log"

```

where the frst python script basically run the the model on a dummy input to register the memory footprint and number of operations (output in _logs/$experiment_name_), while the second starts the actual pretraining. It is possible to follow the training progression by openinig a second terminal and run tensorboard with the following command:

```bash
tensorboard --logdir tensorboard/$experiment_name
```

while it is also possible to look into training logs in the corresponding folder (_logs/$experiment_name_).

Since training may take a while the following command can be useful to launch the scripts in background:

```bash
nohup ./your_script_runner.sh > /location/of/the/output/file.log 2>&1 &
```

## Example Usage

In the directory *notebooks*, it is possible to find a simple example where the model is employed to perform prediction on a test sample. Further information can be found in the brief jupyter notebook.
