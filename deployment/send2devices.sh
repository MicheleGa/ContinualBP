# Send to Raspberry Pi 5
scp ./deployment_component_factory.py ./deployment_maml.py ./DriftDetectors.py ./Proto.py ./run_subject_cli.py ./run_subject.py iris@toaster.local:~/Documents/ContinualBP/deployment/

# Send to Google Pixel 10a
scp -P 8022 ./deployment_component_factory.py ./deployment_maml.py ./DriftDetectors.py ./Proto.py ./run_subject_cli.py ./run_subject.py u0_a365@localhost:/data/data/com.termux/files/home/ContinualBP/deployment/