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
        model_path=relative_to_this_file("../models/cifar10/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx"),
        dataset="cifar10",
        use_normalised_dataset=False,
        epsilon=0.0033,
        img_ids=[0,1,9,15,17,20,25,35,43,45,50,53,54,71,72,77,89,129,131,143,147,149,153,163,165,172,179,181,184,195],
        save_dir=relative_to_this_file("results/CResNetA/verify"),
        python_executable=args.python_executable,
    )  # fmt: skip
