import os
import re
import subprocess
from typing import List, Literal

script_dir = os.path.dirname(os.path.abspath(__file__))

VIRTUAL_ENV_PATH = "/home/yuyi/loris5/PRIMA/venv_3dot8/bin/activate"
"""Path to activate Python virtual env."""

SAVE_DIR = os.path.join(script_dir, "results/M6x256/verify")
"""Directory to save results."""

# Model related hyper-parameters.
DATASET: Literal["mnist", "cifar10"] = "mnist"
assert DATASET in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."
USE_NORMALISED_DATASET: bool = False

MODEL_PATH = os.path.join(script_dir, "../models/mnist/mnist-net_256x6.onnx")
IMG_IDS: List[int] = [0,2,9,17,22,27,33,39,44,53,61,68,78,88,91,113,118,127,132,138,140,141,155,158,164,172,185,188,190,197]
SPARSE_N: int = 50
K: int = 3
S: int = 1
SOLVER_MODE: Literal["original", "sci", "sciplus", "sciall"] = "sciplus"
EPSILON: float = 0.033

model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

def main() -> None:
    for img in IMG_IDS:
        command = f"""
        cd "{script_dir}";
        cd "../tf_verify";
        source "{os.path.abspath(VIRTUAL_ENV_PATH)}";
        python Grena_runone_image.py
            --domain refinepoly
            --GRENA True
            --dataset "{DATASET}"
            --netname "{os.path.abspath(MODEL_PATH)}"
            --output_dir "{os.path.abspath(SAVE_DIR)}"
            --timeout_AR 600
            {
                '' if USE_NORMALISED_DATASET
                else '--mean 0 --std 1' if DATASET == "mnist"
                else '--mean 0 0 0 --std 1 1 1'
            }

            {f'--use_wralu "{SOLVER_MODE}"' if SOLVER_MODE != "original" else ""}
            --epsilon "{EPSILON}"
            --imgid "{img}"
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
