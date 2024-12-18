import os
import pandas as pd
from yaml import safe_load
from tabulate import tabulate

def generate_report(path):
    # Read YAML configuration
    yaml_path = os.path.join(path, "config_snapshot.yaml")
    with open(yaml_path, "r") as yaml_file:
        yaml_data = safe_load(yaml_file)

    # Read metadata
    metadata_path = os.path.join(path, "metadata.txt")
    with open(metadata_path, "r") as metadata_file:
        metadata = metadata_file.read()

    # Read training log
    csv_path = os.path.join(path, "training_log.csv")
    training_log = pd.read_csv(csv_path)

    # Generate the report
    report = {
        "General Configuration": yaml_data.get("parameters", {}),
        "Model Configuration": yaml_data.get("model", {}),
        "Data Generators": yaml_data.get("data_generators", {}),
        "Directories": yaml_data.get("directories", {}),
        "Metadata": metadata.strip(),
        "Training Summary": training_log.describe().to_dict()
    }

    # Generate the one-page report in a table format
    report_summary = []
    for section, content in report.items():
        if isinstance(content, dict):
            for key, value in content.items():
                report_summary.append([section, key, value])
        else:
            report_summary.append([section, "Details", content])

    return tabulate(report_summary, headers=["Section", "Key", "Value"])

# Specify the parent directory containing all logs
logs_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/logs"
output_file = "/Users/jordantaylor/PycharmProjects/gender-modeling/combined_report.txt"

# Combine all reports into one
combined_report = ""
for sub_dir in os.listdir(logs_dir):
    full_path = os.path.join(logs_dir, sub_dir)
    if os.path.isdir(full_path):
        sub_report = generate_report(full_path)
        combined_report += f"Report for {sub_dir}\n{'=' * 40}\n{sub_report}\n\n"

# Save the combined report to a single file
with open(output_file, "w") as report_file:
    report_file.write(combined_report)

print(f"Combined report saved to {output_file}")
