import argparse
import os
from helper import run_verification_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--python-executable", type=str, default="python3", help="Path to Python executable")
    args = parser.parse_args()

    run_verification_experiment(
        model_path=relative_to_this_file("../models/mnist/mnist-net_256x6.onnx"),
        dataset="mnist", 
        use_normalised_dataset=False,
        epsilon=0.033,
        img_ids=[0,2,9,17,22,27,33,39,44,53,61,68,78,88,91,113,118,127,132,138,140,141,155,158,164,172,185,188,190,197],
        save_dir=relative_to_this_file("results/M6x256/verify"),
        python_executable=args.python_executable
    )  # fmt: skip
