import csv
import re
import os
import pickle
import numpy as np
from typing import List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

CUTOFF_THRESHOLD: Optional[float] = 1e-5  # Remove neurons which has < threshold absolute Gurobi improvement
SHOW_OUTLIERS: bool = True  # Whether to show outliers in boxplots
RESULTS_DIR = "/home/shauntan/eran/experiment_scripts/results"
MODEL_NAMES: List[str] = [
    "CConvBig",
    "CResNetA",
    "M6x256",
    "MConvMed",
    "MConvSmall",
]


def main():
    if not os.path.exists("./parsed_results"):
        os.makedirs("./parsed_results")

    for model_name in MODEL_NAMES:
        # verify_csv_path = get_verify_csv_path(RESULTS_DIR, model_name)
        # output_path = f"./parsed_results/{model_name}_verify_summary.csv"
        # parse_verify_result(verify_csv_path, output_path)
        bounds_pkl_path = get_bounds_pkl_path(RESULTS_DIR, model_name)
        bounds = mask_bounds(*load_bounds_results(bounds_pkl_path))
        plot_bounds_diff(model_name, *bounds)
        plot_bounds_improvement(model_name, *bounds)


def get_verify_csv_path(results_dir: str, model_name: str) -> str:
    dir = os.path.join(results_dir, model_name, "verify")
    files = os.listdir(dir)
    matching_files = [f for f in files if re.match(r"^GRENA_result.+\.csv$", f)]
    assert len(matching_files) == 1, f"Expected 1 matching file, but found {len(matching_files)}"
    csv_filename = matching_files[0]
    return os.path.join(dir, csv_filename)


def parse_verify_result(csv_result_path: str, output_path: str) -> None:
    # Extract file name and model name
    file_name = os.path.basename(csv_result_path)
    model_name = re.search(r"^GRENA_result_model=(.+)_eps=", file_name).group(1)  # type: ignore

    # Initialize variables for calculating averages and counts
    total_time = 0
    total_non_unknown_time = 0
    row_count = 0
    non_unknown_count = 0
    unknown_count = 0
    verified_count = 0
    falsified_count = 0
    unknown_time = 0
    verified_time = 0
    falsified_time = 0

    # Process the input CSV and write to output CSV
    with open(csv_result_path, "r") as infile, open(output_path, "w", newline="") as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)

        # Write header to output CSV
        csv_writer.writerow(["img_id", "result", "time"])

        for row in csv_reader:
            if not row or not row[0].startswith(model_name):
                continue

            img_id = int(re.search(r"img (\d+)", row[2]).group(1))  # type: ignore
            time = float(re.search(r"^[0-9.]+", row[5]).group())  # type: ignore
            result = row[6]
            assert result in ["Verified", "Unknown", "Falsified"]

            csv_writer.writerow([img_id, result.lower(), time])

            # Update totals for averages and counts
            total_time += time
            row_count += 1
            if result == "Unknown":
                unknown_count += 1
                unknown_time += time
            elif result == "Verified":
                verified_count += 1
                verified_time += time
                total_non_unknown_time += time
                non_unknown_count += 1
            elif result == "Falsified":
                falsified_count += 1
                falsified_time += time
                total_non_unknown_time += time
                non_unknown_count += 1

        # Calculate averages
        average_time = total_time / row_count if row_count > 0 else 0
        average_non_unknown_time = total_non_unknown_time / non_unknown_count if non_unknown_count > 0 else 0
        avg_unknown_time = unknown_time / unknown_count if unknown_count > 0 else 0
        avg_verified_time = verified_time / verified_count if verified_count > 0 else 0
        avg_falsified_time = falsified_time / falsified_count if falsified_count > 0 else 0

        # Write statistics rows
        csv_writer.writerow(["Num. of unknown:", unknown_count])
        csv_writer.writerow(["Num. of verified:", verified_count])
        csv_writer.writerow(["Num. of falsified:", falsified_count])
        csv_writer.writerow(["Total time:", total_time])
        csv_writer.writerow(["Avg. time:", average_time])
        csv_writer.writerow(["Avg. time of verified/falsified (excludes unknown):", average_non_unknown_time])
        csv_writer.writerow(["Avg. time of unknown:", avg_unknown_time])
        csv_writer.writerow(["Avg. time of verified:", avg_verified_time])
        csv_writer.writerow(["Avg. time of falsified:", avg_falsified_time])

    if row_count != 30:
        print(f"WARNING: {file_name} only has {row_count} rows, which != 30.")


def get_bounds_pkl_path(results_dir: str, model_name: str) -> str:
    model_result_dir = os.path.join(results_dir, model_name, "bounds")
    pkl_files = [f for f in os.listdir(model_result_dir) if f.endswith(".pkl")]
    assert len(pkl_files) == 1, f'Expected exactly one .pkl file, found {len(pkl_files)} in "{model_result_dir}".'
    return os.path.join(model_result_dir, pkl_files[0])


def load_bounds_results(pkl_result_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:  # fmt: skip
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
):
    # Mask for all input neurons, only unstable intermediate neurons, and exclude output neurons.
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


def plot_bounds_diff(
    model_name: str,
    gurobi_flattened_lbs: np.ndarray,
    gurobi_flattened_ubs: np.ndarray,
    tailored_flattened_lbs: np.ndarray,
    tailored_flattened_ubs: np.ndarray,
    ori_flattened_lbs: np.ndarray,
    ori_flattened_ubs: np.ndarray,
) -> None:
    # Calculate the differences
    diff1 = tailored_flattened_ubs - gurobi_flattened_ubs
    diff2 = gurobi_flattened_lbs - tailored_flattened_lbs
    diff3 = ori_flattened_ubs - gurobi_flattened_ubs
    diff4 = gurobi_flattened_lbs - ori_flattened_lbs

    # Plot
    df = pd.DataFrame(
        {
            "Comparison": ["tailored_ubs - gurobi_ubs"] * len(diff1)
            + ["gurobi_lbs - tailored_lbs"] * len(diff2)
            + ["ori_ubs - gurobi_ubs"] * len(diff3)
            + ["gurobi_lbs - ori_lbs"] * len(diff4),
            "Difference": np.concatenate([diff1, diff2, diff3, diff4]),
        }
    )
    plt.figure(figsize=(8, 4))
    sns.boxplot(y="Comparison", x="Difference", data=df, orient="h")
    plt.title(f"Bound Differences\n({model_name})")
    plt.xlabel("Difference")
    plt.tight_layout()
    plt.ylabel("")
    plt.show()

    # Remove differences < CUTOFF_THRESHOLD
    if CUTOFF_THRESHOLD is not None:
        ubs_cutoff_mask = np.abs(diff3) > CUTOFF_THRESHOLD
        lbs_cutoff_mask = np.abs(diff4) > CUTOFF_THRESHOLD
        diff1 = diff1[ubs_cutoff_mask]
        diff2 = diff2[lbs_cutoff_mask]
        diff3 = diff3[ubs_cutoff_mask]
        diff4 = diff4[lbs_cutoff_mask]

        df = pd.DataFrame(
            {
                "Comparison": ["tailored_ubs - gurobi_ubs"] * len(diff1)
                + ["gurobi_lbs - tailored_lbs"] * len(diff2)
                + ["ori_ubs - gurobi_ubs"] * len(diff3)
                + ["gurobi_lbs - ori_lbs"] * len(diff4),
                "Difference": np.concatenate([diff1, diff2, diff3, diff4]),
            }
        )
        plt.figure(figsize=(8, 4))
        sns.boxplot(y="Comparison", x="Difference", data=df, orient="h")
        plt.title(f"Bound Differences ({CUTOFF_THRESHOLD} cutoff)\n({model_name})")
        plt.xlabel("Difference")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()


def plot_bounds_improvement(
    model_name: str,
    gurobi_flattened_lbs: np.ndarray,
    gurobi_flattened_ubs: np.ndarray,
    tailored_flattened_lbs: np.ndarray,
    tailored_flattened_ubs: np.ndarray,
    ori_flattened_lbs: np.ndarray,
    ori_flattened_ubs: np.ndarray,
) -> None:
    # Calculate improvements
    gurobi_improvement = (ori_flattened_ubs - ori_flattened_lbs) - (gurobi_flattened_ubs - gurobi_flattened_lbs)
    our_improvement = (ori_flattened_ubs - ori_flattened_lbs) - (tailored_flattened_ubs - tailored_flattened_lbs)
    df = pd.DataFrame(
        {
            "Method": ["Gurobi"] * len(gurobi_improvement) + ["Our"] * len(our_improvement),
            "Improvement": np.concatenate([gurobi_improvement, our_improvement]),
        }
    )
    plt.figure(figsize=(3, 5))
    sns.boxplot(x="Method", y="Improvement", data=df, showfliers=SHOW_OUTLIERS)
    plt.title(f"Bounds Improvement\n({model_name})")
    plt.ylabel("Improvement")
    plt.show()

    # Remove differences < CUTOFF_THRESHOLD
    if CUTOFF_THRESHOLD is not None:
        mask = np.abs(gurobi_improvement) > CUTOFF_THRESHOLD
        gurobi_improvement = gurobi_improvement[mask]
        our_improvement = our_improvement[mask]

    df = pd.DataFrame(
        {
            "Method": ["Gurobi"] * len(gurobi_improvement) + ["Our"] * len(our_improvement),
            "Improvement": np.concatenate([gurobi_improvement, our_improvement]),
        }
    )
    plt.figure(figsize=(3, 5))
    sns.boxplot(x="Method", y="Improvement", data=df, showfliers=SHOW_OUTLIERS)
    plt.title(f"Bounds Improvement ({CUTOFF_THRESHOLD} cutoff)\n({model_name})")
    plt.ylabel("Improvement")
    plt.show()


if __name__ == "__main__":
    main()
