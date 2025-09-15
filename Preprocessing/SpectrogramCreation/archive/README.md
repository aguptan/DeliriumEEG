## Overview

**What it does**

1. Optionally/Prerequisite: Converts raw .EEG files into per-patient .mat files containing eegStruct using EEG2MAT.m (BioSig-based importer).
2. Loads `.mat` files from a source directory.
3. Iterates through each dataset within `eegStruct`.
4. Preprocesses signals:

   * NaN interpolation and validation
   * Notch filtering at 20, 60, 80 Hz
   * High-pass Butterworth filter at 0.5 Hz (order 4)
   * Common average re-referencing
5. Optional PSD check for a single electrode.
6. Time-windows each dataset at 2, 15, 30, and 120 minutes.
7. Saves per-electrode spectrogram images under a structured output tree.
8. Logs successes, skips, and errors to per-dataset logs and an overall summary.

**Why it exists**

* To produce standardized spectrogram images for downstream model training and evaluation (patient-wise, electrode-wise, and time-window comparisons).
* To ensure reproducible preprocessing with explicit logging.

---

## Requirements

* MATLAB R2019a or newer (tested with modern releases)
* Signal Processing Toolbox (recommended)
* Sufficient disk space for spectrogram images

---

## Directory Layout and Paths

Edit these variables at the top of the script to match your environment:

* `addpath('Location');`
  Folder containing helper functions:

  * `removeNaN`
  * `applyNotchFilter`
  * `applyButter`
  * `reReference`
  * `CaP_PSD`
  * `timeWindowEEG`
  * `saveAllElectrodeSpectrograms`

* `data_dir = 'File_Location';`
  Source directory containing `.mat` files. Each file should contain `eegStruct`.

* `spectrogram_root = 'Location';`
  Root directory where spectrograms and logs will be written.

* `summary_log_path = fullfile(spectrogram_root, 'processing_summary.txt');`
  Global summary log file.


### **Input to EEG2MAT.m**

1. **Raw data root (`raw_data_root`)**

   * A directory containing **patient subfolders**.
   * Each patient folder must contain one or more raw EEG recordings in **`.EEG` format**.
   * Example:

     ```
     RawData/
     ├── Patient01/
     │   ├── file1.EEG
     │   ├── file2.EEG
     ├── Patient02/
     │   ├── session1.EEG
     │   ├── session2.EEG
     ```

2. **BioSig toolbox (`biosig_path`)**

   * Needed to read `.EEG` files (`sopen`, `sread`, `sclose`).
   * Path is validated at runtime.

3. **Output directory (`output_dir`)**

   * The folder where processed `.mat` files will be saved.
   * Created if it doesn’t exist.

---

### **Output of EEG2MAT.m**

1. **Per-patient `.mat` file** written to `output_dir`

   * Filename is based on the patient folder name (e.g., `Patient01.mat`).
   * Contains a single variable: **`eegStruct`**, which is an array of datasets.

2. **Structure of `eegStruct`**
   Each element corresponds to one `.EEG` file inside the patient’s folder.
   Fields:

   * `Filename` → Original `.EEG` filename.
   * `Data` → Numeric matrix of shape `[T x C]` (time samples × channels).
   * `Labels` → Cell array of channel labels read from the EEG header.

   Example layout for `Patient01.mat`:

   ```matlab
   eegStruct(1).Filename = 'file1.EEG'
   eegStruct(1).Data     = [T x C matrix]
   eegStruct(1).Labels   = {'Fp1', 'Fp2', 'C3', ...}

   eegStruct(2).Filename = 'file2.EEG'
   eegStruct(2).Data     = [T x C matrix]
   eegStruct(2).Labels   = {'Fp1', 'Fp2', 'C3', ...}
   ```


## **Expected Input Format for Spectrogram_Creation.m**

Each `.mat` file must contain a variable named `eegStruct`, an array of datasets. For each dataset `eegStruct(dataset_idx)` the script expects:

* `Data`: numeric matrix of shape \[T x C] where T is time samples and C are channels. The script selects channels 1:19 for processing.
* `Labels`: vector or cell array of length >= max(eeg\_channels) providing labels for channels.

If these fields differ in name or structure in your data, adapt the script accordingly where `eegStruct(dataset_idx).Data` and `eegStruct(dataset_idx).Labels` are referenced.

---

## **Expected Output Format for Spectrogram_Creation.m**

For each time window duration W in `{120, 30, 15, 2}` minutes:

```
Spectrograms/
└── Wmin/
    └── <base_filename>_<dataset_idx>/
        ├── <per-electrode spectrograms>.png
        └── <base_filename>_<dataset_idx>_Wmin_log.txt
```

Additionally:

* `processing_summary.txt` at the root of `spectrogram_root` contains a line-by-line summary of datasets processed or skipped with reasons.

---

## Processing Pipeline Details

0. (Optional Seperate File) Upstream Conversion (EEG2MAT.m)

   Use EEG2MAT.m to ingest raw .EEG recordings and write per-patient .mat files that this pipeline consumes:

      `Input:` raw_data_root containing patient subfolders, each with one or more .EEG files.

      `Process:` Iterates patients → reads each .EEG via BioSig (sopen/sread) → assembles an eegStruct array with one entry per file.

      `Output:` Saves <patient_id>.mat under output_dir, each containing eegStruct with fields:

      `Filename:` original .EEG filename

      `Data:` numeric matrix of shape [T x C] (transposed to time-by-channel)

      `Labels:` channel labels from header (hdr.Label)

      `Paths/Setup:`

         raw_data_root – root directory of patient folders

         output_dir – where <patient_id>.mat files are written

         biosig_path – location of the BioSig toolbox (validated at runtime)

         Behavior & Logging: Prints progress for each patient/file, handles file-level errors without stopping the whole run, and writes .mat files with -v7.3 for large data.

         Hand-off: The spectrogram pipeline’s data_dir should point to output_dir from EEG2MAT.m so it can load the produced eegStruct files directly.

1. **Load**

   * Scans `data_dir` for all `.mat` files.
   * For each file, extracts `base_filename`, loads `eegStruct`, and iterates datasets.

2. **Channel Selection**

   * `eeg_channels = 1:19`.
   * Extracts `eeg_data = dat(:, eeg_channels)'` producing \[C x T] for processing.
   * Extracts `selected_labels = labels(eeg_channels)` for naming and metadata.

3. **NaN Handling**

   * `removeNaN(eeg_data)` interpolates or signals too many NaNs.
   * If `nanFlag` is true, the dataset is skipped and logged.

4. **Notch Filtering**

   * `applyNotchFilter` with `notchFreqs = [20, 60, 80]`, Q = 75.
   * Set `plt = 0` to disable PSD plots during notch step.

5. **High-Pass Filtering**

   * `applyButter` with cutoff 0.5 Hz, 4th order.

6. **Re-Referencing**

   * `reReference` applies common average reference.

7. **PSD Check**

   * `CaP_PSD` can generate a PSD for `electrode_idx = 10` if `plt = 1`. By default the script uses `plt = 0`.

8. **Time Windowing**

   * `timeWindowEEG(rreeg_data, fs, window_duration)` splits signals into windows of W minutes.
   * Skips datasets shorter than the requested window.

9. **Spectrogram Export**

   * `saveAllElectrodeSpectrograms(tw_eeg_data, fs, eeg_channels, selected_labels, current_parent_dir, folder_name, window_duration)` writes one or more images per electrode.

10. **Logging**

    * Per-dataset log via `diary` at:
      `Spectrograms3/Wmin/<folder_name>/<folder_name>_Wmin_log.txt`
    * Global summary appended to `processing_summary.txt`.

---

## Configuration

Adjust near the top of the script:

* Sampling frequency: `fs = 200;`
* Channels: `eeg_channels = 1:19;`
* Notch frequencies: `notchFreqs = [20, 60, 80];`
* Notch quality factor: `Q = 75;`
* High-pass cutoff: `hp_cutoff = 0.5;`
* High-pass filter order: `filter_order = 4;`
* PSD electrode index: `electrode_idx = 10;`
* Plotting switch: `plt = 0;`
* Time windows (minutes): `time_windows = sort([2, 15, 30, 120], 'descend');`

---

## How To Run

1. Open MATLAB.
2. Ensure the Functions folder is on path:

   ```matlab
   addpath('Location');
   ```
3. Set `data_dir` and `spectrogram_root` to your drives.
4. Run the script:

   ```matlab
   Spectrogram_Creation  % or press Run in the Editor
   ```
5. Monitor the Command Window for progress. Per-dataset logs are saved automatically.

---

## Logging and Reproducibility

* All status messages for each dataset and time window are written to text logs under the corresponding `Wmin/<folder_name>` directory.
* A global `processing_summary.txt` accumulates person-level outcomes across the entire run.
* To rerun for only failed or skipped datasets, filter by log content in `processing_summary.txt` and selectively reprocess.

---

## Troubleshooting

* **No `.mat` files detected**: Verify `data_dir` and that it contains `.mat` files with `eegStruct`.
* **Missing fields**: Ensure `eegStruct(dataset_idx).Data` and `.Labels` exist and match the expected shapes.
* **Dataset too short**: For small recordings, reduce the time windows in `time_windows`.
* **Out-of-memory or slow performance**: Reduce window sizes, limit channels, or disable plotting (`plt = 0`).
* **Spectrograms not saved**: Confirm `saveAllElectrodeSpectrograms` exists in the Functions path and that `spectrogram_root` is writable.

---

## Notes

* Paths in the script are currently set to local drive letters (E: and D:). Update these to match your system before running.
* If you change sampling frequency, channel list, or filter settings, document the changes and keep a copy of the modified script under version control.

---

