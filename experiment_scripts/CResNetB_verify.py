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
        model_path=relative_to_this_file("../models/cifar10/resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx"),
        dataset="cifar10",
        use_normalised_dataset=False,
        epsilon=0.012,
        img_ids=[0,9,15,17,20,25,35,50,54,71,79,83,87,91,93,96,97,102,109,119,123,129,131,137,147,149,152,163,172,184],
        save_dir=relative_to_this_file("results/CResNetB/verify"),
        python_executable=args.python_executable,
    )  # fmt: skip
