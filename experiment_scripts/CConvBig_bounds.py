import os
from helper import run_bounds_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    run_bounds_experiment(
        model_path=relative_to_this_file("../models/cifar10/convBigRELU__DiffAI.onnx"),
        dataset="cifar10",
        use_normalised_dataset=True,
        epsilon=0.00784313725,
        img_id=80,
        save_dir=relative_to_this_file("results/CConvBig/bounds"),
    )
