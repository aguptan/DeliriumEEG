# Analysis Module

This folder contains scripts and utilities for processing, cleaning, storing, and analyzing results from Leave-One-Out Cross-Validation (LOOCV) experiments on EEG-based delirium classification models. The workflow is organized into **data cleaning**, **container construction**, and **analysis/visualization**.

## File Overview

### 1. **`CleanedJsonFiles.py`**

Utility script for cleaning raw model log outputs (in `.txt` JSON-line format) and saving them into structured `.json` files.

* **Key Features:**

  * Removes redundant fields from epoch entries (e.g., training stats stored in multiple places).
  * Extracts global threshold information (`top_10_thresholds`, `patient_classification_threshold`) from the first epoch for compact storage.
  * Outputs cleaned JSON files grouped by test set and time window.
* **Usage:**

  * Specify `log_folder` (path to raw logs).
  * Script automatically generates cleaned `.json` files with the time window suffix.


### 2. **`unified_container.py`**

Defines the **core data structures** and helper methods to unify fold-level and image-level results into a consistent container for downstream analysis.

#### 🔹 Core Classes

* **`EpochMetrics`** – Training & test stats for a single epoch (losses, accuracies, learning rate, parameters).
* **`PatientPredictionResult`** – Per-patient aggregate predictions:

  * True vs predicted label.
  * Image counts (`positive_images`, `total_images`).
  * Threshold-related metadata (`optimal_threshold`, `top_10_thresholds`, `patient_classification_threshold`).
* **`FoldResult`** – Captures all epoch results for a patient/fold.
* **`ImagePredictionRecord` / `ImagePredictionSet`** – Image-level predictions across epochs for a given patient.
* **`PatientTestSetResult`** – Bundles `FoldResult` and `ImagePredictionSet` for one patient in a test set.

#### 🔹 Unified Container

* **`UnifiedResultsContainer`**:

  * Loads cleaned fold logs (`.json`) and image predictions (`.csv`).
  * Validates consistency between them.
  * Can be saved/loaded as a `.pkl` file for efficient reuse.
  * Organizes results hierarchically:

    ```
    results_by_time_window[time_window][patient_id][test_set] -> PatientTestSetResult
    ```

#### 🔹 Analysis Helper

* **`UnifiedAnalysisHelper`**:
  Provides **Pandas DataFrame interfaces** for exploration:

  * `patient_df()`: Patient-level records (labels, thresholds, predictions).
  * `epoch_df()`: Training/validation metrics across epochs.
  * `image_df()`: Image-level predictions, filterable by patient, filename, or epoch range.
  * Convenience methods like `get_images_for_epoch()` and `get_images_for_patient()`.

This abstraction makes it easy to build visualizations and statistical analyses without manually parsing logs.

### 3. **`loocv_VTS_container.py`**

An earlier version of the unified container, primarily supporting fold and image-level storage.

* Missing newer features like threshold lists (`top_10_thresholds`) and per-patient classification thresholds.
* Retained for backward compatibility with older experimental logs.


### 4. **`loocv_VTS_analysis2.py`**

The **main analysis and visualization script**. It demonstrates how to leverage the unified container and helper to compute metrics and generate figures.

#### 🔹 Workflow

1. **Load Container**

   * Reads in a pre-built `UnifiedResultsContainer` (`.pkl`).
   * Wraps it with `UnifiedAnalysisHelper` for easy DataFrame access.

2. **Compute Metrics**

   * Patient-level adjusted accuracy.
   * Correct vs incorrect image counts.
   * Weighted accuracy aggregated by test set.
   * Optimal threshold values.

3. **Generate Figures**

   * **Figure 1**: Adjusted Accuracy Heatmap — per patient × test set, showing best-epoch accuracy.
   * **Figure 2**: Weighted Accuracy Scatter — per-patient performance with mean lines.
   * **Figure 3**: Threshold vs Accuracy — regression analysis of thresholds and outcomes.
   * **Figure 4**: Confusion Matrices — across test sets, optionally normalized.
   * **Figure 5**: TP/TN/FP/FN Counts — grouped bar plots of classification outcomes.

4. **Output**

   * Interactive plots (via Matplotlib).
   * Can be extended for saving to file or batch-reporting.

**Explanation:**

* **Step A → B**: Clean noisy training logs into compact JSONs.
* **Step B → C**: Load cleaned logs & predictions into a unified container.
* **Step C → D**: Use `UnifiedAnalysisHelper` to access structured results.
* **Step D → E**: Run the main analysis script to generate figures.
* **Step E → F**: Produce interpretable outputs for reporting/publication.

---

## Typical Workflow

1. **Clean raw log files**

   ```bash
   python CleanedJsonFiles.py
   ```

   → Produces cleaned `.json` files for each time window.

2. **Build unified container**

   ```python
   from unified_container import UnifiedResultsContainer
   container = UnifiedResultsContainer()
   container.load_fold_logs(time_window="30min", log_dir="path/to/cleaned/logs")
   container.load_image_predictions(time_window="30min", pred_dir="path/to/predictions")
   container.save("unified_container.pkl")
   ```

3. **Run analysis**

   ```bash
   python loocv_VTS_analysis2.py
   ```

   → Generates and displays Figures 1–5.

---

## Requirements

* Python 3.8+
* Libraries: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`