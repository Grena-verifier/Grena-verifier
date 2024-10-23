import os
from helper import run_verification_experiment


def relative_to_this_file(path: str) -> str:
    """Set path to be relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


if __name__ == "__main__":
    run_verification_experiment(
        model_path=relative_to_this_file("../models/mnist/convMedGRELU__Point.onnx"),
        dataset="mnist",
        use_normalised_dataset=False,
        epsilon=0.1,
        img_ids=[0,1,2,3,4,5,7,9,10,11,12,13,14,17,18,19,20,21,22,23,24,26,27,28,29,30,35,37,40,47],
        save_dir=relative_to_this_file("results/MConvMed/verify"),
    )  # fmt: skip
