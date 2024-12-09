import argparse
import os
from helper import run_bounds_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--python-executable", type=str, default="python3", help="Path to Python executable")
    args = parser.parse_args()

    run_bounds_experiment(
        model_display_name="MConvSmall",
        model_path=relative_to_this_file("../models/mnist/convSmallRELU__Point.onnx"),
        dataset="mnist",
        use_normalised_dataset=False,
        epsilon=0.11,
        img_id=75,
        save_dir=relative_to_this_file("results/MConvSmall/bounds"),
        python_executable=args.python_executable,
    )
