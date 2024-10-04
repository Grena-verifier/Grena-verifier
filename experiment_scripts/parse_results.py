import csv
import re
import os


def main():
    RESULTS_DIR = "/home/shauntan/eran/experiment_scripts/results"
    MODEL_NAMES: list[str] = ["CConvBig", "CResNetA", "M6x256", "MConvMed", "MConvSmall"]

    if not os.path.exists("./parsed_results"):
        os.makedirs("./parsed_results")

    for model_name in MODEL_NAMES:
        verify_csv_path = get_verify_csv_path(RESULTS_DIR, model_name)
        output_path = f"./parsed_results/{model_name}_verify_summary.csv"
        parse_verify_result(verify_csv_path, output_path)


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

            csv_writer.writerow([img_id, result, time])

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


if __name__ == "__main__":
    main()
