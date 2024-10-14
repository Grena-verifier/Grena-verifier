import os
from helper import run_verification_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    run_verification_experiment(
        model_path=relative_to_this_file("../models/cifar10/convBigRELU__DiffAI.onnx"),
        dataset="cifar10",
        use_normalised_dataset=True,
        epsilon=0.00784313725,
        img_ids=[0,12,13,17,20,32,33,47,49,55,65,71,80,88,91,97,101,102,103,104,115,119,130,132,145,155,163,165,174,192],
        save_dir=relative_to_this_file("results/CConvBig/verify"),
    )  # fmt: skip
