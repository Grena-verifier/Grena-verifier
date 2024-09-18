import os
import re
import subprocess
from typing import Literal

VIRTUAL_ENV_PATH = "/home/yuyi/loris5/PRIMA/venv_3dot8/bin/activate"
"""Path to activate Python virtual env."""

SAVE_DIR = "experiments/MConvMed/bounds"
"""Directory to save results."""

# Model related hyper-parameters.
DATASET = "mnist"
MODEL_PATH = "../models/mnist/convMedGRELU__Point.onnx"
IMG_ID: int = 2
SPARSE_N: int = 50
K: int = 3
S: int = 1
SOLVER_MODE: Literal["original", "sci", "sciplus", "sciall"] = "sciplus"
EPSILON: float = 0.1

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
