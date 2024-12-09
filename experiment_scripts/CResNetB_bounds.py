import os
from helper import run_bounds_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    run_bounds_experiment(
        model_display_name="CResNetB",
        model_path=relative_to_this_file("../models/cifar10/resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx"),
        dataset="cifar10",
        use_normalised_dataset=False,
        epsilon=0.012,
        img_id=79,
        save_dir=relative_to_this_file("results/CResNetB/bounds"),
    )
