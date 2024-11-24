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
        model_path=relative_to_this_file("../models/mnist/convSmallRELU__Point.onnx"),
        dataset="mnist",
        use_normalised_dataset=False,
        epsilon=0.11,
        img_ids=[0,1,2,3,4,5,6,7,11,12,15,20,21,34,35,40,41,47,48,62,75,79,83,86,92,101,138,139,160,184],
        save_dir=relative_to_this_file("results/MConvSmall/verify"),
        python_executable=args.python_executable,
    )  # fmt: skip
