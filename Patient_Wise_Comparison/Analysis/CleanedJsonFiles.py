import os
import json
from pathlib import Path

# --- CONFIGURATION ---
log_folder = Path(r"E:\Amal\DeliriumProject\ModelRun\Output\VaryingTestSets_8.2.25\30min\logs")
timewindow = log_folder.parent.name  # e.g., "120min"
output_suffix = f"_{timewindow}.json"

# --- HELPER FUNCTIONS ---

def remove_redundant_fields(epoch_data, global_thresholds):
    """Clean a single epoch dictionary by removing redundant fields."""
    cleaned = {
        "epoch": epoch_data["epoch"],
        "image_n_parameters": epoch_data["image_n_parameters"],
        "train_stats": epoch_data["train_stats"],
        "test_results_by_group": {}
    }

    for group, group_data in epoch_data["test_results_by_group"].items():
        # Clean test_stats
        test_stats = group_data["test_stats"].copy()
        for key in ["train_lr", "train_loss", "epoch", "image_n_parameters"]:
            test_stats.pop(key, None)

        # Clean patient_predictions
        patient_preds = []
        for pred in group_data["patient_predictions"]:
            pred_clean = pred.copy()
            for key in ["top_10_thresholds", "patient_classification_threshold"]:
                pred_clean.pop(key, None)
            patient_preds.append(pred_clean)

        cleaned["test_results_by_group"][group] = {
            "test_stats": test_stats,
            "patient_predictions": patient_preds
        }

    return cleaned

def extract_global_thresholds(epoch_data):
    """Pull shared thresholds from first patient in first group."""
    for group_data in epoch_data["test_results_by_group"].values():
        patient_preds = group_data.get("patient_predictions", [])
        if patient_preds:
            first_pred = patient_preds[0]
            return {
                "top_10": first_pred.get("top_10_thresholds", []),
                "classification_threshold": first_pred.get("patient_classification_threshold", None)
            }
    return None

# --- MAIN PROCESS ---

txt_files = list(log_folder.glob("*.txt"))
print(f"Found {len(txt_files)} .txt files in {log_folder}...")

for txt_file in txt_files:
    with open(txt_file, "r") as f:
        lines = f.readlines()

    print(f"Processing: {txt_file.name} ({len(lines)} epochs)")

    # Load all epoch JSONs
    epoch_data_list = [json.loads(line.strip()) for line in lines]

    # Extract shared thresholds from first epoch
    thresholds = extract_global_thresholds(epoch_data_list[0])

    # Clean all epoch entries
    cleaned_epochs = [remove_redundant_fields(epoch, thresholds) for epoch in epoch_data_list]

    # New structure
    cleaned_data = {
        "thresholds": thresholds,
        "epochs": cleaned_epochs
    }

    # Output file
    out_path = txt_file.with_name(txt_file.stem + output_suffix)
    with open(out_path, "w") as out_f:
        json.dump(cleaned_data, out_f, indent=2)

    print(f" → Saved cleaned file: {out_path.name}")

print("✅ Cleaning complete.")
