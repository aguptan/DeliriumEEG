# Integrated EEG Spectrogram Pipeline

## Overview

This pipeline standardizes preprocessing of raw EEG recordings into spectrogram images for downstream analysis and model training.

### **Stages**

1. **Sequential Import (EEG → MAT)**

   * Converts raw `.EEG` recordings into per-patient `.mat` files.
   * Uses [BioSig toolbox](https://biosig.sourceforge.net/) (`sopen`, `sread`, `sclose`) for file access.
   * Saves one `.mat` per patient, each containing an `eegStruct`.

2. **Parallel Preprocessing + Spectrogram Export**

   * Loads `.mat` files and iterates datasets × time windows.
   * Runs in parallel via MATLAB `parfor`.
   * Steps include NaN handling, notch & high-pass filtering, common average re-referencing, PSD check, time-windowing, and spectrogram generation.
   * Logs every dataset and writes a global summary.

---

## Requirements

* MATLAB R2019a or newer (tested with recent releases)
* Parallel Computing Toolbox (for Stage 2)
* Signal Processing Toolbox (recommended)
* [BioSig Toolbox](https://biosig.sourceforge.net/) (for Stage 1, EEG import)
* Sufficient disk space for spectrogram images

---

## Directory Setup

At the top of the script, edit:

```matlab
RAW_DATA_ROOT   = 'Path_to_raw_EEG_root';     % Root folder with patient subfolders containing .EEG files
EEG2MAT_DIR     = 'Path_to_save_EEG2MAT';    % Output folder for per-patient .mat files
SPECTROGRAM_DIR = 'Path_for_Spectrograms';   % Final spectrogram output root
BIOSIG_PATH     = 'Path_to_BioSig/t200_FileAccess'; % BioSig toolbox location
FUNC_PATH       = 'Path_to_Functions';       % Folder with preprocessing functions
```

---

## Input and Output

### **Stage 1 Input**

* Root directory with patient subfolders, each containing `.EEG` recordings.
* Example:

  ```
  RawData/
  ├── 1. JH/
  │   ├── recording1.EEG
  │   ├── recording2.EEG
  ├── 2. AB/
  │   ├── session1.EEG
  ```

### **Stage 1 Output**

* Per-patient `.mat` file saved under `EEG2MAT_DIR`.
* Each `.mat` contains `eegStruct`:

  ```matlab
  eegStruct(1).Filename = 'recording1.EEG';
  eegStruct(1).Data     = [channels x samples];
  eegStruct(1).Labels   = {'Fp1','Fp2','C3',...};
  ```

### **Stage 2 Output**

* Spectrogram images organized by time window and dataset:

  ```
  Spectrograms/
  ├── 30min/
  │   ├── JH_1_1/
  │   │   ├── Fp1.png
  │   │   ├── Fp2.png
  │   │   └── JH_1_1_30min_log.txt
  ├── 60min/
  │   └── ...
  ```
* `MASTER_LOG_<timestamp>.mat` with metadata and results.
* `processing_summary_<timestamp>.txt` global summary.

---

## Processing Details

* **Channel selection**: electrodes `1:19`
* **NaN handling**: interpolate or skip dataset
* **Filters**:

  * Notch @ 20, 60, 80 Hz (Q=75)
  * High-pass @ 0.5 Hz, order 4
* **Re-reference**: common average
* **Time windows**: `[30, 60, 120, 150, 180]` minutes (configurable)
* **Spectrogram export**: one per electrode per window
* **Parallelization**: each dataset × time window is a separate parallel job

---

## How to Run

1. Install BioSig and set `BIOSIG_PATH`.
2. Place your helper functions (`removeNaN2`, `applyNotchFilter`, `applyButter`, `reReference`, `CaP_PSD`, `timeWindowEEG2`, `saveAllElectrodeSpectrograms5`) in `FUNC_PATH`.
3. Edit the paths at the top of the script.
4. Run the script in MATLAB:

   ```matlab
   IntegratedPipeline  % or press Run in the Editor
   ```
5. Monitor the console and check logs in the spectrogram output folder.

---

## Logging & Reproducibility

* Each dataset produces a per-window log.
* A global summary file is created with successes, skips, and errors.
* A `.mat` `MASTER_LOG` contains metadata for structured analysis.

---

## Troubleshooting

* **No `.EEG` files found** → Check `RAW_DATA_ROOT`.
* **BioSig not found** → Verify `BIOSIG_PATH` points to `/biosig/t200_FileAccess`.
* **No `.mat` files in Stage 2** → Run Stage 1 first.
* **Dataset skipped** → Too many NaNs or too short for requested time window.
* **Slow or memory-heavy** → Reduce time windows, disable plotting (`plt=0`), or limit channels.

