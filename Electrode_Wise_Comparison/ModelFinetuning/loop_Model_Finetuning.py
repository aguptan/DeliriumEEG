import os
import subprocess
import time
import pandas as pd
import json
import csv

# === Base directory containing all time-duration subfolders ===
base_dir = "/media/enver/Seagate Portable Drive/Delerium-EEG"
spectrograms_dir = os.path.join(base_dir, "Spectrograms2")
output_base_dir = os.path.join(base_dir, "checkpoints")

# === Paths to pretrained model and fine-tune script ===
pretrained_model_path = os.path.join(base_dir, "DL Code", "GMML-ECoG-alldata", "channel wise", "best_ckpt_ep310.pth")
finetune_script_path = os.path.join(base_dir, "DL Code", "GMML-ECoG-alldata", "channel wise", "finetune_EEG.py")

# === Discover time-duration folders ===
time_durations = [
    d for d in os.listdir(spectrograms_dir)
    if os.path.isdir(os.path.join(spectrograms_dir, d))
]

overall_start_time = time.time()

for time_duration in time_durations:
    print(f"\n=== Starting time duration: {time_duration} ===")

    input_dir = os.path.join(spectrograms_dir, time_duration, "ByElectrode")
    output_dir = os.path.join(output_base_dir, time_duration)
    os.makedirs(output_dir, exist_ok=True)

    summary = []
    time_window_start = time.time()
    summary_csv_path = os.path.join(output_dir, "fine_tuning_summary.csv")
    
    print("All electrode folders detected:")

    all_electrodes = [
        f for f in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, f))
    ]
    # electrodes = ["C3", "Fz"]
    for electrode in all_electrodes: 
        print(f"Processing electrode: {electrode}")
        electrode_start = time.time()

        data_location = os.path.join(input_dir, electrode)
        train_folder = os.path.join(data_location, "train")

        if not os.path.isdir(train_folder):
            print(f"'train' folder not found inside {electrode}, skipping.")
            summary.append({
                "Electrode": electrode,
                "Status": "missing_train",
                "Time (s)": 0.0,
                "Best Epoch": "",
                "Best AUC": "",
                "AUPR": "",
                "TPR": "",
                "TNR": "",
                "Log Path": ""
            })
            try:
                pd.DataFrame(summary).to_csv(summary_csv_path, index=False)
            except Exception as e:
                print(f" Failed to write summary for {electrode}: {e}")
            continue

        electrode_output_dir = os.path.join(output_dir, electrode)
        os.makedirs(electrode_output_dir, exist_ok=True)

        command = [
            "python", finetune_script_path,
            "--data_location", data_location,
            "--finetune", pretrained_model_path,
            "--output_dir", electrode_output_dir
        ]

        try:

            subprocess.run(command, check=True)
            elapsed = time.time() - electrode_start
            print(f" Finished fine-tuning for {electrode} in {elapsed:.1f} seconds")
            status = "success"
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - electrode_start
            print(f" Fine-tuning failed for {electrode}: {e}")
            status = "failed"

        # === Parse log file ===
        log_file = os.path.join(electrode_output_dir, "log_test.txt")
        best_auc = ""
        best_epoch = ""
        best_aupr = ""
        best_tpr = ""
        best_tnr = ""

        if os.path.isfile(log_file):
            try:
                with open(log_file, "r") as f:
                    max_auc = -1
                    best_metrics = {}
                    for line in f:
                        try:
                            log = json.loads(line.strip())
                            auc = log.get("EnsembleAUC")
                            epoch = log.get("epoch")
                            if auc is not None and epoch is not None and auc > max_auc:
                                max_auc = auc
                                best_metrics = {
                                    "epoch": epoch,
                                    "auc": auc,
                                    "aupr": log.get("EnsembleAUPR"),
                                    "tpr": log.get("TPR"),
                                    "tnr": log.get("TNR")
                                }
                        except json.JSONDecodeError:
                            continue
                    if best_metrics:
                        best_epoch = best_metrics["epoch"]
                        best_auc = round(best_metrics["auc"], 4)
                        best_aupr = round(best_metrics["aupr"], 4) if best_metrics["aupr"] is not None else ""
                        best_tpr = round(best_metrics["tpr"], 4) if best_metrics["tpr"] is not None else ""
                        best_tnr = round(best_metrics["tnr"], 4) if best_metrics["tnr"] is not None else ""
            except Exception as e:
                print(f" Could not parse log for {electrode}: {e}")

        summary.append({
            "Electrode": electrode,
            "Status": status,
            "Time (s)": round(elapsed, 2),
            "Best Epoch": best_epoch,
            "Best AUC": best_auc,
            "AUPR": best_aupr,
            "TPR": best_tpr,
            "TNR": best_tnr,
            "Log Path": log_file if os.path.isfile(log_file) else ""
        })

        try:
            pd.DataFrame(summary).to_csv(summary_csv_path, index=False)
        except Exception as e:
            print(f" Failed to write summary after {electrode}: {e}")

    elapsed_window = time.time() - time_window_start
    print(f"\n Completed: {time_duration} in {elapsed_window / 60:.2f} minutes")

    try:
        print(pd.DataFrame(summary).to_string(index=False))
    except Exception as e:
        print(f" Could not print summary: {e}")

total_elapsed = time.time() - overall_start_time
print("\n All fine-tuning complete.")
print(f" Total time: {total_elapsed / 60:.2f} minutes")