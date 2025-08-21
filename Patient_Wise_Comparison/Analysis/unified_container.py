from typing import Dict, List
from dataclasses import dataclass, field
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
    top_10_thresholds: List[float] = field(default_factory=list)
    patient_classification_threshold: float = 0.5


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
            # Extract patient ID as the part after "log_test_"
            parts = log_file.stem.split("_")  # ['log', 'test', '1', '30min']
            patient_id = int(parts[2])        # '1'

            with open(log_file, "r") as f:
                log_data = json.load(f)
            top_10_thresholds = log_data.get("thresholds", {}).get("top_10", [])
            classification_threshold = log_data.get("thresholds", {}).get("classification_threshold", 0.5)

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
                            optimal_threshold=patient_entry["optimal_threshold"],
                            top_10_thresholds=top_10_thresholds,
                            patient_classification_threshold=classification_threshold
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

        pattern = r"patient_(\d+)__fold_\d+_0\.5_(.+)_image_predictions"

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
        
import pandas as pd
from typing import Optional, Tuple

class UnifiedAnalysisHelper:
    def __init__(self, container):
        self.container = container
        self._cached_patient_dfs = {}
        self._cached_epoch_dfs = {}
        self._cached_image_dfs = {}

    def patient_df(self, time_window: Optional[str] = None, epoch_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        key = (time_window, epoch_range)
        if key in self._cached_patient_dfs:
            return self._cached_patient_dfs[key]

        records = []
        windows = [time_window] if time_window else self.container.results_by_time_window.keys()

        for tw in windows:
            for pid, test_sets in self.container.results_by_time_window[tw].items():
                for test_set, result in test_sets.items():
                    fold = result.fold_result
                    if fold is None:
                        continue
                    for epoch in fold.epochs:
                        if epoch_range and not (epoch_range[0] <= epoch.epoch <= epoch_range[1]):
                            continue
                        p = epoch.patient_prediction
                        records.append({
                            "time_window": tw,
                            "test_set": test_set,
                            "fold_id": fold.fold_id,
                            "epoch": epoch.epoch,
                            "true_label": p.true_label,
                            "pred_label": p.pred_label,
                            "optimal_threshold": p.optimal_threshold,
                            "positive_images": p.positive_images,
                            "total_images": p.total_images,
                            "optimal_threshold": p.optimal_threshold,
                            "patient_classification_threshold": getattr(p, "patient_classification_threshold", None),
                            "top_10_thresholds": p.top_10_thresholds,
                            "test_acc1": epoch.test_acc1,
                            "test_loss": epoch.test_loss
                        })

        df = pd.DataFrame(records)
        self._cached_patient_dfs[key] = df
        return df

    def epoch_df(self, time_window: Optional[str] = None, epoch_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        key = (time_window, epoch_range)
        if key in self._cached_epoch_dfs:
            return self._cached_epoch_dfs[key]

        rows = []
        windows = [time_window] if time_window else self.container.results_by_time_window.keys()

        for tw in windows:
            for pid, test_sets in self.container.results_by_time_window[tw].items():
                for test_set, result in test_sets.items():
                    fold = result.fold_result
                    if fold is None:
                        continue
                    for epoch in fold.epochs:
                        if epoch_range and not (epoch_range[0] <= epoch.epoch <= epoch_range[1]):
                            continue
                        rows.append({
                            "time_window": tw,
                            "test_set": test_set,
                            "fold_id": fold.fold_id,
                            "epoch": epoch.epoch,
                            "train_loss": epoch.train_loss,
                            "test_loss": epoch.test_loss,
                            "test_acc1": epoch.test_acc1,
                            "test_acc5": epoch.test_acc5,
                            "lr": epoch.lr,
                            "n_params": epoch.n_params,
                            "true_label": epoch.patient_prediction.true_label,
                            "pred_label": epoch.patient_prediction.pred_label,
                            "optimal_threshold": epoch.patient_prediction.optimal_threshold
                        })

        df = pd.DataFrame(rows)
        self._cached_epoch_dfs[key] = df
        return df

    def image_df(
        self,
        time_window: Optional[str] = None,
        epoch_range: Optional[Tuple[int, int]] = None,
        filename_contains: Optional[str] = None
    ) -> pd.DataFrame:
        key = (time_window, epoch_range, filename_contains)
        if key in self._cached_image_dfs:
            return self._cached_image_dfs[key]

        rows = []
        windows = [time_window] if time_window else self.container.results_by_time_window.keys()

        for tw in windows:
            for pid, test_sets in self.container.results_by_time_window[tw].items():
                for test_set, result in test_sets.items():
                    img_set = result.image_predictions
                    if img_set is None:
                        continue
                    for r in img_set.predictions:
                        if epoch_range and not (epoch_range[0] <= r.epoch <= epoch_range[1]):
                            continue
                        if filename_contains and filename_contains not in r.filename:
                            continue
                        rows.append({
                            "time_window": tw,
                            "test_set": test_set,
                            "patient_id": pid,
                            "epoch": r.epoch,
                            "filename": r.filename,
                            "true_label": r.true_label,
                            "pred_label": r.pred_label,
                            "probability": r.probability
                        })

        df = pd.DataFrame(rows)
        self._cached_image_dfs[key] = df
        return df

    def get_images_for_epoch(self, time_window: str, epoch_number: int) -> pd.DataFrame:
        return self.image_df(time_window, epoch_range=(epoch_number, epoch_number))

    def get_images_for_patient(self, time_window: str, patient_id: int) -> pd.DataFrame:
        df = self.image_df(time_window)
        return df[df["patient_id"] == patient_id]

    def summary(self) -> str:
        lines = []
        for tw, patient_map in self.container.results_by_time_window.items():
            for test_set in set(ts for sets in patient_map.values() for ts in sets):
                n_patients = sum(1 for p in patient_map.values() if test_set in p)
                lines.append(f"🕒 Time Window: {tw} | Test Set: {test_set} | Patients: {n_patients}")
        return "\n".join(lines)
    
    

