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
        model_path=relative_to_this_file("../models/cifar10/convMedGRELU__PGDK_w_0.0078.onnx"),
        dataset="cifar10",
        use_normalised_dataset=True,
        epsilon=0.006,
        img_ids=[4,7,28,45,50,59,60,64,70,72,74,88,95,103,105,110,115,135,137,142,176,180,193,220,230,326,332,348,379,434],
        save_dir=relative_to_this_file("results/CConvMed/verify"),
        python_executable=args.python_executable,
    )  # fmt: skip
