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
        model_display_name="CResNetA",
        model_path=relative_to_this_file("../models/cifar10/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx"),
        dataset="cifar10",
        use_normalised_dataset=False,
        epsilon=0.0033,
        img_id=89,
        save_dir=relative_to_this_file("results/CResNetA/bounds"),
        python_executable=args.python_executable,
    )
