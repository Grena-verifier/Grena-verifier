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
        model_path=relative_to_this_file("../models/cifar10/resnet_4b.onnx"),
        dataset="cifar10",
        use_normalised_dataset=False,
        epsilon=0.0042,
        img_ids=[0,1,7,17,20,34,39,45,50,53,54,72,78,79,83,85,87,91,97,119,131,159,181,213,233,247,293,383,426,479],
        save_dir=relative_to_this_file("results/CResNet4B/verify"),
        python_executable=args.python_executable,
    )  # fmt: skip
