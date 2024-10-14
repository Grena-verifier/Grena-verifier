import os
import re
import subprocess
from typing import List, Literal

VIRTUAL_ENV_PATH = "/home/yuyi/loris5/PRIMA/venv_3dot8/bin/activate"  # Path to activate Python virtual env.

# ERAN related hyperparams.
SPARSE_N: int = 50
K: int = 3
S: int = 1
SOLVER_MODE: Literal["original", "sci", "sciplus", "sciall"] = "sciplus"


script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this file.


def run_bounds_experiment(
    model_path: str,
    dataset: Literal["mnist", "cifar10"],
    use_normalised_dataset: bool,
    epsilon: float,
    img_id: int,
    save_dir: str,
) -> None:
    assert dataset in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    save_file_name = (
        model_name + f"_epsilon={epsilon}" + f"_imgid={img_id}" + "_domain=refinepoly" + f"_solver={SOLVER_MODE}" + ".pkl"
    )

    command = f"""
    source "{VIRTUAL_ENV_PATH}";
    cd "{os.path.join(script_dir, "../tf_verify")}";
    mkdir -p "{save_dir}";
    python Grena_runone_image.py
        --domain refinepoly
        --dataset "{dataset}"
        --netname "{model_path}"
        --output_dir "{save_dir}"
        --bounds_save_filename "{save_file_name}"
        --epsilon "{epsilon}"
        --imgid "{img_id}"
        {
            '' if use_normalised_dataset
            else '--mean 0 --std 1' if dataset == "mnist"
            else '--mean 0 0 0 --std 1 1 1'
        }

        {f'--use_wralu "{SOLVER_MODE}"' if SOLVER_MODE != "original" else ""}
        --sparse_n "{SPARSE_N}"
        --k "{K}"
        --s "{S}"
        >> "{os.path.join(save_dir, "terminal.log")}" 2>&1
    ;
    """
    command = re.sub(r"\n\s*", " ", command).strip()
    print(f"Running:\n{command}")
    subprocess.run(command, shell=True, executable="/bin/bash")


def run_verification_experiment(
    model_path: str,
    dataset: Literal["mnist", "cifar10"],
    use_normalised_dataset: bool,
    epsilon: float,
    img_ids: List[int],
    save_dir: str,
) -> None:
    assert dataset in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."

    for img in img_ids:
        command = f"""
        source "{VIRTUAL_ENV_PATH}";
        cd "{os.path.join(script_dir, "../tf_verify")}";
        mkdir -p "{save_dir}";
        python Grena_runone_image.py
            --domain refinepoly
            --GRENA True
            --dataset "{dataset}"
            --netname "{model_path}"
            --output_dir "{save_dir}"
            --epsilon "{epsilon}"
            --imgid "{img}"
            --timeout_AR 600
            {
                '' if use_normalised_dataset
                else '--mean 0 --std 1' if dataset == "mnist"
                else '--mean 0 0 0 --std 1 1 1'
            }

            {f'--use_wralu "{SOLVER_MODE}"' if SOLVER_MODE != "original" else ""}
            --sparse_n "{SPARSE_N}"
            --k "{K}"
            --s "{S}"
            >> "{os.path.join(save_dir, "terminal.log")}" 2>&1
        ;
        """
        command = re.sub(r"\n\s*", " ", command).strip()
        print(f"Running:\n{command}")
        subprocess.run(command, shell=True, executable="/bin/bash")
