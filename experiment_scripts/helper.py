import csv
import os
import pickle
import re
import subprocess
import sys
from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import TypeAlias

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


# RNG seed for reproducibility. Set to None to disable seeding.
SEED: Optional[int] = 42

# ERAN related hyperparams.
SPARSE_N: int = 50
K: int = 3
S: int = 1
SOLVER_MODE: Literal["original", "sci", "sciplus", "sciall"] = "sciplus"

# Bounds-experiment boxplot params.
ModelName: TypeAlias = Literal[
    "CConvMed",
    "CResNet4B",
    "CResNetA",
    "CResNetB",
    "M6x256",
    "MConvBig",
    "MConvMed",
    "MConvSmall",
]

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this file.


def run_bounds_experiment(
    model_display_name: ModelName,
    model_path: str,
    dataset: Literal["mnist", "cifar10"],
    use_normalised_dataset: bool,
    epsilon: float,
    img_id: int,
    save_dir: str,
) -> None:
    generate_bounds_result(model_path, dataset, use_normalised_dataset, epsilon, img_id, save_dir)
    extract_runtimes_into_csv(save_dir, model_display_name)
    bounds_pkl_path = get_bounds_pkl_path(save_dir)
    bounds = mask_bounds(*load_bounds_results(bounds_pkl_path))
    cutoff_threshold = 1e-20
    plot_bounds_improvement(
        model_display_name,
        *bounds,
        img_save_path=os.path.join(save_dir, f"RESULT_bounds_improvement_plot.jpg"),
        cutoff_threshold=cutoff_threshold,
    )


def run_verification_experiment(
    model_path: str,
    dataset: Literal["mnist", "cifar10"],
    use_normalised_dataset: bool,
    epsilon: float,
    img_ids: List[int],
    save_dir: str,
) -> None:
    assert dataset in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."
    results_path = os.path.join(
        save_dir, "RESULT_GRENA_verification.csv"
    )  # defined in `tf_verify/Grena_runone_image.py`

    write_results_csv_header(results_path)
    for img_id in img_ids:
        verify_image_using_grena(model_path, dataset, use_normalised_dataset, epsilon, img_id, save_dir)
    append_results_summary(results_path)


# ===============================================================================
#                     Bounds experiment's helper functions
# ===============================================================================
def generate_bounds_result(
    model_path: str,
    dataset: Literal["mnist", "cifar10"],
    use_normalised_dataset: bool,
    epsilon: float,
    img_id: int,
    save_dir: str,
) -> None:
    assert dataset in ["mnist", "cifar10"], "This script isn't designed for datasets other than MNIST and CIFAR10."
    log_path = os.path.join(save_dir, "terminal.log")

    command = f"""
    cd "{os.path.join(script_dir, "../tf_verify")}";
    mkdir -p "{save_dir}";
    '{sys.executable}' Grena_runone_image.py
        --domain refinepoly
        --dataset "{dataset}"
        --netname "{model_path}"
        --output_dir "{save_dir}"
        --epsilon "{epsilon}"
        --imgid "{img_id}"
        {'' if SEED is None else f"--seed {SEED}"}
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


def extract_runtimes_into_csv(save_dir: str, model_display_name: str) -> None:
    """Extract Gurobi and Tailored solver runtimes from log file and save to CSV."""
    with open(f"{save_dir}/log", "r") as f:
        log_content = f.read()

    gurobi_match = re.search(r"Gurobi: (\d+\.\d+)", log_content)
    tailored_match = re.search(r"Tailored solver: (\d+\.\d+)", log_content)
    assert gurobi_match is not None and tailored_match is not None
    gurobi_time_str = gurobi_match.group(1)
    tailored_time_str = tailored_match.group(1)

    with open(f"{save_dir}/RESULT_solver_runtimes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Network", "Gurobi runtime (seconds)", "Tailored solver runtime (seconds)"])
        writer.writerow([model_display_name, gurobi_time_str, tailored_time_str])


def get_bounds_pkl_path(save_dir: str) -> str:
    """Get the pickle file path for a bounds result."""
    pkl_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]
    assert len(pkl_files) == 1, f'Expected exactly one .pkl file, found {len(pkl_files)} in "{save_dir}".'
    return os.path.join(save_dir, pkl_files[0])


def load_bounds_results(pkl_result_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:  # fmt: skip
    """Load pickle file for a bounds result as flattened HWC-format numpy arrays."""
    with open(pkl_result_path, "rb") as file:
        _ = pickle.load(file)

    gurobi_flattened_lbs = np.concatenate([x.flatten() for x in _["gurobi_lbs"]])
    gurobi_flattened_ubs = np.concatenate([x.flatten() for x in _["gurobi_ubs"]])
    ori_flattened_lbs = np.concatenate([x.flatten() for x in _["IOIL_lbs"]])
    ori_flattened_ubs = np.concatenate([x.flatten() for x in _["IOIL_ubs"]])

    # Convert tailored bounds from CHW -> HWC if they're are 3D.
    # Gurobi/Original bounds are in HWC, but tailored bounds uses Pytorch's CHW format.
    tailored_lbs = _["tailored_lbs"]
    tailored_ubs = _["tailored_ubs"]
    for i, (lbs, ubs) in enumerate(zip(tailored_lbs, tailored_ubs)):
        if lbs.ndim == 3:
            tailored_lbs[i] = np.transpose(lbs, (1, 2, 0))
            tailored_ubs[i] = np.transpose(ubs, (1, 2, 0))
    tailored_flattened_lbs = np.concatenate([x.flatten() for x in tailored_lbs])
    tailored_flattened_ubs = np.concatenate([x.flatten() for x in tailored_ubs])

    assert gurobi_flattened_lbs.shape == gurobi_flattened_ubs.shape == tailored_flattened_lbs.shape == tailored_flattened_ubs.shape == ori_flattened_lbs.shape == ori_flattened_ubs.shape  # fmt: skip
    return (
        gurobi_flattened_lbs,
        gurobi_flattened_ubs,
        tailored_flattened_lbs,
        tailored_flattened_ubs,
        ori_flattened_lbs,
        ori_flattened_ubs,
        _["IOIL_lbs"],
        _["IOIL_ubs"],
    )


def mask_bounds(
    gurobi_flattened_lbs: np.ndarray,
    gurobi_flattened_ubs: np.ndarray,
    tailored_flattened_lbs: np.ndarray,
    tailored_flattened_ubs: np.ndarray,
    ori_flattened_lbs: np.ndarray,
    ori_flattened_ubs: np.ndarray,
    ori_lbs: List[np.ndarray],
    ori_ubs: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask for all input neurons, only unstable intermediate neurons, and exclude output neurons."""
    num_layers = len(ori_lbs)
    intermediate_unstable_masks = np.concatenate(
        [(ori_lbs[i] < 0) & (ori_ubs[i] > 0) for i in range(1, num_layers - 1)]
    )
    mask = np.concatenate(
        [
            np.full_like(ori_lbs[0], True, dtype=bool),
            intermediate_unstable_masks,
            np.full_like(ori_lbs[-1], False, dtype=bool),
        ]
    )
    return (
        gurobi_flattened_lbs[mask],
        gurobi_flattened_ubs[mask],
        tailored_flattened_lbs[mask],
        tailored_flattened_ubs[mask],
        ori_flattened_lbs[mask],
        ori_flattened_ubs[mask],
    )


def plot_bounds_improvement(
    model_display_name: str,
    gurobi_flattened_lbs: np.ndarray,
    gurobi_flattened_ubs: np.ndarray,
    tailored_flattened_lbs: np.ndarray,
    tailored_flattened_ubs: np.ndarray,
    ori_flattened_lbs: np.ndarray,
    ori_flattened_ubs: np.ndarray,
    img_save_path: Optional[str] = None,
    cutoff_threshold: Optional[float] = None,
    max_exponent: Union[int, Literal["auto"]] = "auto",
    min_exponent: Union[int, Literal["auto"]] = "auto",
    num_bins: int = 40,
) -> None:
    """
    Visualizes the improvement in bounds between Gurobi solver and our tailored solver using a histogram.

    Creates a symmetric log-scale histogram comparing bound improvements (reductions in bound width)
    between Gurobi and tailored solvers relative to original bounds. The visualization splits
    improvements into a negative (bounds worsening) region on the left and positive (bounds
    improvement) regions on the right.

    Parameters:
        model_display_name (str): Name of the model to display in the plot title
        gurobi_flattened_lbs (np.ndarray): Lower bounds from Gurobi
        gurobi_flattened_ubs (np.ndarray): Upper bounds from Gurobi
        tailored_flattened_lbs (np.ndarray): Lower bounds from our tailored solver
        tailored_flattened_ubs (np.ndarray): Upper bounds from our tailored solver
        ori_flattened_lbs (np.ndarray): Original lower bounds
        ori_flattened_ubs (np.ndarray): Original upper bounds
        img_save_path (Optional[str]): If provided, save plot as image. If not, show the plot.
        cutoff_threshold (Optional[float]): If provided, excludes neurons with Gurobi-improvements \
            below this threshold (default: None)
        max_exponent (Union[int, Literal["auto"]]): Max exponent for log-scale bins. If "auto", \
            use data's max (default: "auto")
        min_exponent (Union[int, Literal["auto"]]): Min exponent for log-scale bins. If "auto", \
            use 5th percentile of non-zero Gurobi-improvements after cutoff (default: "auto")
        num_bins (int): Number of log bins to use in each half of the histogram (default: 50)
    """
    # Calculate improvements (reduction in bound width)
    gurobi_improvement = (ori_flattened_ubs - ori_flattened_lbs) - (gurobi_flattened_ubs - gurobi_flattened_lbs)  # type: ignore
    our_improvement = (ori_flattened_ubs - ori_flattened_lbs) - (tailored_flattened_ubs - tailored_flattened_lbs)  # type: ignore

    # Remove neurons with Gurobi-improvements below cutoff threshold
    if cutoff_threshold is not None:
        mask = np.abs(gurobi_improvement) > cutoff_threshold
        gurobi_improvement = gurobi_improvement[mask]
        our_improvement = our_improvement[mask]

    # If max_exponent == "auto", dynamically scale `max_exponent` based on the max of the data
    if max_exponent == "auto":
        max_abs = np.max(np.abs([gurobi_improvement, our_improvement]))
        max_exponent_in_arrays = int(np.floor(np.log10(max_abs))) if max_abs != 0 else 0
        max_exponent = max_exponent_in_arrays + 1
        assert np.all(np.abs([gurobi_improvement, our_improvement]) < 10**max_exponent)

    # If min_exponent == "auto", dynamically scale `min_exponent` to the 5th-percentile of Gurobi's data (excluding 0s)
    if min_exponent == "auto":
        gurobi_non_zeros = gurobi_improvement[gurobi_improvement != 0]
        gurobi_percentile_5 = np.percentile(gurobi_non_zeros, 5)
        assert gurobi_percentile_5 != 0
        gurobi_percentile_5_exponent = int(np.floor(np.log10(gurobi_percentile_5)))
        min_exponent = gurobi_percentile_5_exponent

    min_magnitude = 10**min_exponent  # Smallest bin
    max_magnitude = 10**max_exponent  # Largest bin

    # Clamp values to bin range (min_magnitude to max_magnitude)
    gurobi_improvement = np.where(gurobi_improvement == 0, min_magnitude * 1.001, gurobi_improvement)
    gurobi_improvement = np.where(
        np.abs(gurobi_improvement) < min_magnitude,
        np.sign(gurobi_improvement) * min_magnitude * 1.001,  # slightly more to ensure it falls in biggest bin
        gurobi_improvement,
    )
    gurobi_improvement = np.where(
        np.abs(gurobi_improvement) > max_magnitude,
        np.sign(gurobi_improvement) * max_magnitude * 0.999,  # slightly less to ensure it falls in smallest bin
        gurobi_improvement,
    )

    our_improvement = np.where(our_improvement == 0, min_magnitude * 1.001, our_improvement)
    our_improvement = np.where(
        np.abs(our_improvement) < min_magnitude,
        np.sign(our_improvement) * min_magnitude * 1.001,  # slightly more to ensure it falls in biggest bin
        our_improvement,
    )
    our_improvement = np.where(
        np.abs(our_improvement) > max_magnitude,
        np.sign(our_improvement) * max_magnitude * 0.999,  # slightly less to ensure it falls in smallest bin
        our_improvement,
    )

    # Create figure with two subplots side by side with no spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0)

    # Separate positive and negative improvements
    pos_gurobi = gurobi_improvement[gurobi_improvement >= 0]
    pos_our = our_improvement[our_improvement >= 0]
    neg_gurobi = -gurobi_improvement[gurobi_improvement < 0]  # Take absolute value
    neg_our = -our_improvement[our_improvement < 0]  # Take absolute value

    # Create fixed bins for both positive and negative improvements
    fixed_bins = np.logspace(min_exponent, max_exponent, num_bins)

    # Plot negative improvements on the left
    neg_counts_gurobi = ax1.hist(neg_gurobi, bins=fixed_bins, alpha=0.7, label="Gurobi")[0]
    neg_counts_our = ax1.hist(neg_our, bins=fixed_bins, alpha=0.7, label="Our")[0]

    # Plot positive improvements on the right
    pos_counts_gurobi = ax2.hist(pos_gurobi, bins=fixed_bins, alpha=0.7, label="Gurobi")[0]
    pos_counts_our = ax2.hist(pos_our, bins=fixed_bins, alpha=0.7, label="Our")[0]

    # Set the same y-axis limits on both negative and positive plots
    max_count = max(
        max(
            pos_counts_gurobi.max() if len(pos_counts_gurobi) > 0 else 0,
            pos_counts_our.max() if len(pos_counts_our) > 0 else 0,
        ),
        max(
            neg_counts_gurobi.max() if len(neg_counts_gurobi) > 0 else 0,
            neg_counts_our.max() if len(neg_counts_our) > 0 else 0,
        ),
    )
    ax1.set_ylim(0, max_count * 1.1)  # Add 10% padding
    ax2.set_ylim(0, max_count * 1.1)

    # Set x-axis to use log-scale
    ax1.set_xscale("log")
    ax2.set_xscale("log")

    # Fix x-axis for both plots to be [min_magnitude, max_magnitude]
    ax2.set_xlim(min_magnitude, max_magnitude)
    ax1.set_xlim(max_magnitude, min_magnitude)  # Reversed limits

    # Visually replace smallest bin ticker with 0
    def pos_formatter(x: float, p: int) -> str:
        if min_magnitude * 0.999 <= x and x <= min_magnitude * 1.001:  # approx min_magnitude
            return "0"
        power = int(np.log10(x))
        return f"10$^{{{power}}}$"

    def neg_formatter(x: float, p: int) -> str:
        if min_magnitude * 0.999 <= x and x <= min_magnitude * 1.001:  # approx min_magnitude
            return ""  # don't label, to avoid overlap with positive 0 ticker
        power = int(np.log10(x))
        return f"10$^{{{power}}}$"

    ax1.xaxis.set_major_formatter(FuncFormatter(neg_formatter))
    ax2.xaxis.set_major_formatter(FuncFormatter(pos_formatter))

    # Other misc. plot configurations
    ax1.set_xlabel("Negative Improvement (log-scale)")
    ax1.set_ylabel("Count")
    ax1.set_title("Bounds worsening")
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    ax2.set_xlabel("Improvement (log-scale)")
    ax2.set_title("Bounds improvement")
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Put negative and positive improvement graphs side-by-side
    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    plt.suptitle(
        f"Bounds Improvement Distribution ({model_display_name}) ({cutoff_threshold if cutoff_threshold is not None else 'no'} cutoff)"
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    if img_save_path is not None:
        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
        plt.savefig(img_save_path)
        plt.close()
    else:
        plt.show()


# ===============================================================================
#                  Verification experiment's helper functions
# ===============================================================================
def write_results_csv_header(results_path: str) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "", "", "GRENA", ""])
        csv_writer.writerow(["Image index", "Dataset / Network", "Epsilon", "Result", "Time (seconds)"])


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
) -> None:
    log_path = os.path.join(save_dir, "terminal.log")
    command = f"""
    cd "{os.path.join(script_dir, "../tf_verify")}";
    mkdir -p "{save_dir}";
    '{sys.executable}' Grena_runone_image.py
        --domain refinepoly
        --GRENA True
        --dataset "{dataset}"
        --netname "{model_path}"
        --output_dir "{save_dir}"
        --epsilon "{epsilon}"
        --imgid "{img_id}"
        --timeout_AR 600
        {'' if SEED is None else f"--seed {SEED}"}
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
