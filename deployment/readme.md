# Note

# PI Deployment
For deployment on teh Raspberry PI, the *deployment_personalizer.py* script is not required. Also, the import in the deployment scritps must be modified adequately to run adequately with the actual path on the Pi. It is also possible and recommended to add the laptop ssh key to avoid entering the password and enable the python scripts to work smoothly.

# Pixel Deployment
> - Install Android Debug Bridge on Ubuntu 22.04. Then connect the phone via USB after having enabled the developer options and enabled USB debugging on the pixel.
> - Install termux on the phone from [here](https://f-droid.org/en/packages/com.termux/)  (specifically, press on the Download APK button under the latest specific version, 29th of May for the time of this project). It is possible to install the apk via adb:

```bash 
adb install /path/to/the/termux/file.apk
```

> - On the phone, open a termux session and update packeages, then install essential packages

```bash
pkg update && pkg upgrade -y
pkg install -y openssh git wget curl proot-distro
```

> - Setup SSH inm termux. After restart it is important to *manually start* sshd (inside termux by typing *sshd*) otherwise ssh from laptop won't work. Then it is also possible and recommended to add the laptop ssh key to avoid entering the password and enable the python scripts to work smoothly.

```bash
passwd          # set a password for SSH login
sshd            # starts SSH daemon on port 8022
```

> - On the laptop it is now possible to connect to the phone via SSH

```bash
adb forward tcp:8022 tcp:8022
ssh -p 8022 $(adb shell whoami | tr -d '\r')@localhost
```

or (if you know your user name on termux)

```bash
adb forward tcp:8022 tcp:8022
ssh -p 8022 u0_a365@localhost
```

while for scp (note the upper case P):

```bash
adb forward tcp:8022 tcp:8022
ssh -P 8022 u0_a365@localhost:/path/to/dest/folder
```

> -  Now it is possible to run commands from the laptop and execute them on the phone. The next step is setting up the python + pytorch environment to run scripts:

```bash
proot-distro install ubuntu
```

> -  Termux automatically install the latest version of Ubuntu (25, which antivaely supports python 3.13 which is not suitable for torch 2.5.1 used in this project) installing the 22.04 (the preferred one for this project) is a bit more complicated, so we proceed by installing pyenv and switch to python 3.10. From inside Ubuntu 25

```bash
apt update && apt upgrade -y
apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev git
curl https://pyenv.run | bash
```

> - Add pyenv to ~/.bashrc

```bash
cat >> ~/.bashrc << 'EOF'

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
```

> -  And, as always, immediately

```bash
source ~/.bashrc
```

> - Switch to python 3.10 (compiling it takes a while, > 15 minutes):

```bash
pyenv --version   
pyenv install 3.10.14
pyenv global 3.10.14
python --version
```

> - Now it is possible to create a virtualenvironment and proceed with the installaiton of the python packages required for deployment (see the README.md file)

For automating the experiments run on the phone, the following bash script can be useful:

```bash
cat > ~/run_continualbp.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash

DATA_FILE=$1
WEIGHTS_FILE=$2
CONFIG_FILE=$3
RESULTS_FILE=$4

proot-distro login ubuntu -- bash -c "
cd ~/ContinualBP/deployment && \
source venv/bin/activate && \
python run_subject_cli.py \
${DATA_FILE} \
${WEIGHTS_FILE} \
${CONFIG_FILE} \
${RESULTS_FILE}
"
EOF
chmod +x ~/run_continualbp.sh
```

> - As for the Pi, the import in the deployment scritps must be modified adequately to run adequately with the actual path inside the pixel.  