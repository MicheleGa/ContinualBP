import sys
import pickle
 
# Allow imports relative to the project root (adjust if needed)
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from run_subject import run_subject   # already present on the Pi
 # Ensure Torch uses only three threads: cpu from 0-2 cores
import torch
torch.set_num_threads(3)


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python run_subject_cli.py "
            "<data_pkl> <weights_pt> <config_pkl> <results_pkl>",
            file=sys.stderr,
        )
        sys.exit(1)
 
    data_path, weights_path, config_path, results_path = sys.argv[1:5]
    print(f"[Deployment On-Device] data    : {data_path}")
    print(f"[Deployment On-Device] weights : {weights_path}")
    print(f"[Deployment On-Device] config  : {config_path}")
    print(f"[Deployment On-Device] results : {results_path}")
 
    baseline_outputs, baseline_targets, profiling_report = run_subject(
        data_path, weights_path, config_path
    )
 
    with open(results_path, "wb") as f:
        pickle.dump((baseline_outputs, baseline_targets, profiling_report), f)
 
    print(f"[Deployment On-Device] Results saved to {results_path}")
 
 
if __name__ == "__main__":
    main()