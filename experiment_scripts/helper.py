import csv
import os
import re
import subprocess
from typing import List, Literal


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
    python_executable: str = "python3",  # Path to python executable
) -> None:
    assert dataset in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."
    log_path = os.path.join(save_dir, "terminal.log")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    save_file_name = (
        model_name
        + f"_epsilon={epsilon}"
        + f"_imgid={img_id}"
        + "_domain=refinepoly"
        + f"_solver={SOLVER_MODE}"
        + ".pkl"
    )

    command = f"""
    cd "{os.path.join(script_dir, "../tf_verify")}";
    mkdir -p "{save_dir}";
    '{python_executable}' Grena_runone_image.py
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
        >> "{log_path}" 2>&1
    ;
    """
    command = re.sub(r"\n\s*", " ", command).strip()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"Running:\n{command}\n")
    subprocess.run(command, shell=True, executable="/bin/bash")


def run_verification_experiment(
    model_path: str,
    dataset: Literal["mnist", "cifar10"],
    use_normalised_dataset: bool,
    epsilon: float,
    img_ids: List[int],
    save_dir: str,
    python_executable: str = "python3",  # Path to python executable
) -> None:
    assert dataset in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."
    results_path = os.path.join(save_dir, "GRENA_verification_result.csv")

    write_results_csv_header(results_path)
    for img_id in img_ids:
        verify_image_using_grena(
            model_path, dataset, use_normalised_dataset, epsilon, img_id, save_dir, python_executable
        )
    append_results_summary(results_path)


def write_results_csv_header(results_path: str) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "", "", "GRENA", ""])
        csv_writer.writerow(["Image index", "Network", "Epsilon", "Result", "Time"])


def append_results_summary(results_path: str) -> None:
    data = []
    with open(results_path, "r") as f:
        # Skip the two header lines
        next(f)
        next(f)

        reader = csv.reader(f)
        for row in reader:
            if not row:  # Skip empty lines
                continue
            result = row[3]  # Result column
            time = float(row[4])  # Time column
            data.append((result, time))

    # Compute statistics
    unknown_times = [t for r, t in data if r == "unknown"]
    verified_times = [t for r, t in data if r == "verified"]
    falsified_times = [t for r, t in data if r == "falsified"]

    total_num = len(data)
    num_unknown = len(unknown_times)
    num_verified = len(verified_times)
    num_falsified = len(falsified_times)

    total_time = sum(t for _, t in data)
    avg_time = total_time / total_num if total_num > 0 else -1

    # Average times for each category
    verified_falsified_times = verified_times + falsified_times
    avg_time_verified_falsified = (
        sum(verified_falsified_times) / len(verified_falsified_times) if num_verified + num_falsified > 0 else -1
    )
    avg_time_unknown = sum(unknown_times) / num_unknown if num_unknown > 0 else -1
    avg_time_verified = sum(verified_times) / num_verified if num_verified > 0 else -1
    avg_time_falsified = sum(falsified_times) / num_falsified if num_falsified > 0 else -1

    # Prepare summary lines
    summary_rows = [
        ["Total num. of rows:", total_num],
        ["Num. of unknown:", num_unknown],
        ["Num. of verified:", num_verified],
        ["Num. of falsified:", num_falsified],
        ["Total time:", total_time],
        ["Avg. time:", avg_time],
        ["Avg. time of verified/falsified (excludes unknown):", avg_time_verified_falsified],
        ["Avg. time of unknown:", avg_time_unknown],
        ["Avg. time of verified:", avg_time_verified],
        ["Avg. time of falsified:", avg_time_falsified],
    ]

    # Append summary to the file
    with open(results_path, "a") as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)


def verify_image_using_grena(
    model_path: str,
    dataset: Literal["mnist", "cifar10"],
    use_normalised_dataset: bool,
    epsilon: float,
    img_id: int,
    save_dir: str,
    python_executable: str,
) -> None:
    log_path = os.path.join(save_dir, "terminal.log")
    command = f"""
    cd "{os.path.join(script_dir, "../tf_verify")}";
    mkdir -p "{save_dir}";
    '{python_executable}' Grena_runone_image.py
        --domain refinepoly
        --GRENA True
        --dataset "{dataset}"
        --netname "{model_path}"
        --output_dir "{save_dir}"
        --epsilon "{epsilon}"
        --imgid "{img_id}"
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
        >> "{log_path}" 2>&1
    ;
    """
    command = re.sub(r"\n\s*", " ", command).strip()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"Running:\n{command}\n")
    subprocess.run(command, shell=True, executable="/bin/bash")
