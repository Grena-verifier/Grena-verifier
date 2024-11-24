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
        model_path=relative_to_this_file("../models/mnist/convBigRELU__DiffAI.onnx"),
        dataset="mnist",
        use_normalised_dataset=False,
        epsilon=0.313,
        img_ids=[2,9,24,46,70,81,174,182,183,187,223,235,265,295,296,310,311,326,340,363,386,439,458,466,481,522,524,545,587,591],
        save_dir=relative_to_this_file("results/MConvBig/verify"),
        python_executable=args.python_executable,
    )  # fmt: skip
