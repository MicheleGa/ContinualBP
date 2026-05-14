import sys
import pickle
 
# Allow imports relative to the project root (adjust if needed)
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from run_subject import run_subject   # already present on the Pi
 
 
def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python run_subject_cli.py "
            "<data_pkl> <weights_pt> <config_pkl> <results_pkl>",
            file=sys.stderr,
        )
        sys.exit(1)
 
    data_path, weights_path, config_path, results_path = sys.argv[1:5]
 
    print(f"[Pi] data    : {data_path}")
    print(f"[Pi] weights : {weights_path}")
    print(f"[Pi] config  : {config_path}")
    print(f"[Pi] results : {results_path}")
 
    baseline_outputs, baseline_targets, profiling_report = run_subject(
        data_path, weights_path, config_path
    )
 
    with open(results_path, "wb") as f:
        pickle.dump((baseline_outputs, baseline_targets, profiling_report), f)
 
    print(f"[Pi] Results saved to {results_path}")
 
 
if __name__ == "__main__":
    main()