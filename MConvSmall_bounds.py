import os
import re
import subprocess
from typing import List, Literal
from typing_extensions import TypeAlias

VIRTUAL_ENV_PATH = "/home/yuyi/loris5/PRIMA/venv_3dot8/bin/activate"
"""Path to activate Python virtual env."""

SAVE_DIR = "experiment_mconvsmall_bounds"
"""Directory to save results."""

# Model related hyper-parameters.
DATASET = "mnist"
MODEL_PATH = "models/mnist/convSmallRELU__Point.onnx"
IMG_ID: int = 76
SPARSE_N: int = 50
K: int = 3
S: int = 1
SolverModes: TypeAlias = Literal["original", "sci", "sciplus", "sciall"]
K_RELU_SOLVER_MODES: List[SolverModes] = ["sciplus"]
EPSILONS: List[float] = [0.11]

model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

def main() -> None:
    for solver in K_RELU_SOLVER_MODES:
        for eps in EPSILONS:
            save_file_name = (
                model_name + f"_epsilon={eps}" + f"_imgid={IMG_ID}" + "_domain=refinepoly" + ".pkl"
            )

            command = f"""
            source "{os.path.abspath(VIRTUAL_ENV_PATH)}";
            cd {os.path.abspath("tf_verify")};
            python Grena_runone_image.py
                --domain refinepoly
                --dataset "{DATASET}"
                --netname "{os.path.abspath(MODEL_PATH)}"
                --output_dir "{os.path.abspath(os.path.join(SAVE_DIR, solver))}"
                --bounds_save_filename "{save_file_name}"

                {f'--use_wralu "{solver}"' if solver != "original" else ""}
                --epsilon "{eps}"
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
