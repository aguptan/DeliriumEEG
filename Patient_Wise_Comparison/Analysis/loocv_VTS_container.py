from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
import json
import csv
import re
import pickle

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    test_acc1: float
    test_acc5: float
    lr: float
    n_params: int
    patient_prediction: 'PatientPredictionResult'

@dataclass
class PatientPredictionResult:
    patient_id: int
    true_label: int
    pred_label: int
    positive_images: int
    total_images: int
    optimal_threshold: float

@dataclass
class FoldResult:
    fold_id: int
    patient_id: int
    true_label: int
    epochs: List[EpochMetrics]

@dataclass
class ImagePredictionRecord:
    epoch: int
    filename: str
    true_label: int
    pred_label: int
    probability: float

@dataclass
class ImagePredictionSet:
    patient_id: int
    predictions: List[ImagePredictionRecord]

@dataclass
class PatientTestSetResult:
    fold_result: FoldResult = None
    image_predictions: ImagePredictionSet = None

class UnifiedResultsContainer:
    def __init__(self):
        self.results_by_time_window: Dict[str, Dict[int, Dict[str, PatientTestSetResult]]] = {}

    def load_fold_logs(self, time_window: str, log_dir: str, verbose: bool = True):
        tw_results = self.results_by_time_window.setdefault(time_window, {})
        log_files = sorted(Path(log_dir).glob("log_test_*.json"))

        for log_file in log_files:
            patient_id = int(log_file.stem.split("_")[-1])
            with open(log_file, "r") as f:
                log_data = json.load(f)

            for test_set_name in log_data["epochs"][0]["test_results_by_group"]:
                epochs = []
                for epoch_entry in log_data["epochs"]:
                    group_result = epoch_entry["test_results_by_group"][test_set_name]
                    patient_entry = group_result["patient_predictions"][0]

                    epoch = EpochMetrics(
                        epoch=epoch_entry["epoch"],
                        train_loss=epoch_entry["train_stats"]["loss"],
                        test_loss=group_result["test_stats"]["test_loss"],
                        test_acc1=group_result["test_stats"]["test_acc1"],
                        test_acc5=group_result["test_stats"]["test_acc5"],
                        lr=epoch_entry["train_stats"]["lr"],
                        n_params=epoch_entry["image_n_parameters"],
                        patient_prediction=PatientPredictionResult(
                            patient_id=patient_entry["patient_id"],
                            true_label=patient_entry["true_label"],
                            pred_label=patient_entry["pred_label"],
                            positive_images=patient_entry["positive_images"],
                            total_images=patient_entry["total_images"],
                            optimal_threshold=patient_entry["optimal_threshold"]
                        )
                    )
                    epochs.append(epoch)

                tw_results.setdefault(patient_id, {}).setdefault(test_set_name, PatientTestSetResult()).fold_result = FoldResult(
                    fold_id=patient_id,
                    patient_id=patient_id,
                    true_label=epochs[0].patient_prediction.true_label,
                    epochs=epochs
                )

                if verbose:
                    print(f"✅ Fold {patient_id} - {test_set_name} ({time_window}): {len(epochs)} epochs")

    def load_image_predictions(self, time_window: str, pred_dir: str, verbose: bool = True):
        tw_results = self.results_by_time_window.setdefault(time_window, {})
        prediction_files = sorted(Path(pred_dir).glob("*.csv"))

        pattern = r"patient_(\d+)__fold_\d+_0\.5_([^_]+(?:_[^_]+)*)_image_predictions\.csv"

        for pred_file in prediction_files:
            match = re.match(pattern, pred_file.stem)
            if not match:
                continue

            patient_id = int(match.group(1))
            test_set_name = match.group(2)

            with open(pred_file, "r") as f:
                lines = f.readlines()

            header_index = next(i for i, line in enumerate(lines) if line.strip().startswith("epoch"))
            reader = csv.DictReader(lines[header_index:])

            records = [
                ImagePredictionRecord(
                    epoch=int(row["epoch"]),
                    filename=row["filename"],
                    true_label=int(row["true_label"]),
                    pred_label=int(row["pred_label"]),
                    probability=float(row["probability"])
                )
                for row in reader
            ]

            tw_results.setdefault(patient_id, {}).setdefault(test_set_name, PatientTestSetResult()).image_predictions = ImagePredictionSet(
                patient_id=patient_id,
                predictions=records
            )

            if verbose:
                print(f"📷 Patient {patient_id} - {test_set_name} ({time_window}): {len(records)} image predictions")

    def validate_consistency(self, time_window: str, verbose: bool = True):
        missing = []
        for pid, test_sets in self.results_by_time_window[time_window].items():
            expected = set(test_sets.keys())
            for ts in expected:
                result = test_sets[ts]
                if result.fold_result is None or result.image_predictions is None:
                    missing.append((pid, ts))
        if missing:
            for pid, ts in missing:
                print(f"⚠️ Incomplete result for patient {pid}, test set {ts} in {time_window}")
        elif verbose:
            print(f"✅ All results consistent for {time_window}")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"💾 Saved results container to {path}")

    @staticmethod
    def load(path: str) -> "UnifiedResultsContainer":
        with open(path, "rb") as f:
            print(f"📂 Loaded results container from {path}")
            return pickle.load(f)
