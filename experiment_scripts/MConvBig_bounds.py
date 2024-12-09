import os
from helper import run_bounds_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    run_bounds_experiment(
        model_display_name="MConvBig",
        model_path=relative_to_this_file("../models/mnist/convBigRELU__DiffAI.onnx"),
        dataset="mnist",
        use_normalised_dataset=False,
        epsilon=0.313,
        img_id=174,
        save_dir=relative_to_this_file("results/MConvBig/bounds"),
    )
