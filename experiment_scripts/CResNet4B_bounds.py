import os
from helper import run_bounds_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    run_bounds_experiment(
        model_display_name="CResNet4B",
        model_path=relative_to_this_file("../models/cifar10/resnet_4b.onnx"),
        dataset="cifar10",
        use_normalised_dataset=False,
        epsilon=0.0042,
        img_id=83,
        save_dir=relative_to_this_file("results/CResNet4B/bounds"),
    )
