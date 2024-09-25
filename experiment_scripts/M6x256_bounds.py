import os
import re
import subprocess
from typing import Literal

VIRTUAL_ENV_PATH = "/home/yuyi/loris5/PRIMA/venv_3dot8/bin/activate"
"""Path to activate Python virtual env."""

SAVE_DIR = "results/M6x256/bounds"
"""Directory to save results."""

# Model related hyper-parameters.
DATASET: Literal["mnist", "cifar10"] = "mnist"
assert DATASET in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."
USE_NORMALISED_DATASET: bool = False

MODEL_PATH = "../models/mnist/mnist-net_256x6.onnx"
IMG_ID: int = 9
SPARSE_N: int = 50
K: int = 3
S: int = 1
SOLVER_MODE: Literal["original", "sci", "sciplus", "sciall"] = "sciplus"
EPSILON: float = 0.033

model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

def main() -> None:
    save_file_name = (
        model_name + f"_epsilon={EPSILON}" + f"_imgid={IMG_ID}" + "_domain=refinepoly" + f"_solver={SOLVER_MODE}" + ".pkl"
    )

    command = f"""
    source "{os.path.abspath(VIRTUAL_ENV_PATH)}";
    cd {os.path.abspath("../tf_verify")};
    python Grena_runone_image.py
        --domain refinepoly
        --dataset "{DATASET}"
        --netname "{os.path.abspath(MODEL_PATH)}"
        --output_dir "{os.path.abspath(SAVE_DIR)}"
        --bounds_save_filename "{save_file_name}"
        {
            '' if USE_NORMALISED_DATASET
            else '--mean 0 --std 1' if DATASET == "mnist"
            else '--mean 0 0 0 --std 1 1 1'
        }

        {f'--use_wralu "{SOLVER_MODE}"' if SOLVER_MODE != "original" else ""}
        --epsilon "{EPSILON}"
        --imgid "{IMG_ID}"
        --sparse_n "{SPARSE_N}"
        --k "{K}"
        --s "{S}"
    ;
    """
    command = re.sub(r"\n\s*", " ", command).strip()
    print(f"Running:\n{command}")
    subprocess.run(command, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    main()
