import os
import re
import subprocess
from typing import List, Literal

VIRTUAL_ENV_PATH = "/home/yuyi/loris5/PRIMA/venv_3dot8/bin/activate"
"""Path to activate Python virtual env."""

SAVE_DIR = "results/M6x256/verify"
"""Directory to save results."""

# Model related hyper-parameters.
DATASET: Literal["mnist", "cifar10"] = "mnist"
assert DATASET in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."
USE_NORMALISED_DATASET: bool = False

MODEL_PATH = "../models/mnist/mnist-net_256x6.onnx"
IMG_IDS: List[int] = [1,3,10,18,23,28,34,40,45,54,62,69,79,89,92,114,119,128,133,139,141,142,156,159,165,173,186,189,191,198]
IMG_IDS = [i - 1 for i in IMG_IDS]  # Above IDs uses 1-index, ERAN uses 0-index. This converts it to 0-index.
SPARSE_N: int = 50
K: int = 3
S: int = 1
SOLVER_MODE: Literal["original", "sci", "sciplus", "sciall"] = "sciplus"
EPSILON: float = 0.033

model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

def main() -> None:
    for img in IMG_IDS:
        command = f"""
        source "{os.path.abspath(VIRTUAL_ENV_PATH)}";
        cd {os.path.abspath("../tf_verify")};
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
