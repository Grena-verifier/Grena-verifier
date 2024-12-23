import os
from helper import run_bounds_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    run_bounds_experiment(
        model_display_name="M6x256",
        model_path=relative_to_this_file("../models/mnist/mnist-net_256x6.onnx"),
        dataset="mnist",
        use_normalised_dataset=False,
        epsilon=0.033,
        img_id=9,
        save_dir=relative_to_this_file("results/M6x256/bounds"),
    )
