import os
import re
import subprocess
from typing import List, Literal

VIRTUAL_ENV_PATH = "/home/yuyi/loris5/PRIMA/venv_3dot8/bin/activate"
"""Path to activate Python virtual env."""

SAVE_DIR = "experiments/MConvSmall/verify"
"""Directory to save results."""

# Model related hyper-parameters.
DATASET = "mnist"
MODEL_PATH = "../models/mnist/convSmallRELU__Point.onnx"
IMG_IDS: List[int] = [1,2,3,4,5,6,7,8,12,13,16,21,22,26,35,36,41,42,48,49,50,76,80,84,93,102,139,140,161,185]
SPARSE_N: int = 50
K: int = 3
S: int = 1
SOLVER_MODE: Literal["original", "sci", "sciplus", "sciall"] = "sciplus"
EPSILON: float = 0.11

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
