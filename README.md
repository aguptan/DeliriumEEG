# DeliriumProject – Vision Transformer for EEG Analysis

This repository contains code and resources for a **Vision Transformer (ViT)-based framework** to analyze EEG spectrograms for **delirium detection**. The project integrates preprocessing pipelines, patient- and electrode-wise model finetuning, and detailed post-hoc analysis.

---

## Repository Structure
DeliriumProjectGithub
├── Electrode_Wise_Comparison
│   └── ModelFinetuning
│       ├── README_ElectrodeWiseComparison.md
│       ├── VisionTransformerModel
│       │   ├── ECOG90S_dataloader.py
│       │   ├── GMML.jpg
│       │   ├── GMMLenv_fixed.yml
│       │   ├── README.md
│       │   ├── finetune_EEG.py
│       │   ├── requirements.txt
│       │   ├── utils.py
│       │   └── vision_transformer.py
│       └── loop_Model_Finetuning.py
├── Patient_Wise_Comparison
│   ├── Analysis
│   │   ├── Graphs & Scripts (confusion matrices, heatmaps, accuracy curves, etc.)
│   │   ├── README_analysis.md
│   │   ├── loocv_VTS_analysis2.py
│   │   ├── loocv_VTS_container.py
│   │   └── unified_container.py
│   └── ModelFinetuning
│       ├── LOO_patientwise_finetuning.py
│       ├── MultiGPURun.py
│       ├── README_PatientWise.MD
│       └── VisionTransformerModel
│           ├── dataloader, utils, and ViT definition
│           ├── GMMLenv_fixed.yml
│           └── requirements.txt
└── Preprocessing
    ├── FileMoving
    │   ├── MTFolderManagement.py
    │   └── c2PNG_w2CSV.py
    └── SpectrogramCreation
        ├── EEG2MAT.m
        ├── Functions (filtering, referencing, spectrogram generation, etc.)
        ├── README_Spectrogram.md
        └── Spectrogram_Creation.m

## Main Components

### 1. Preprocessing

* MATLAB and Python scripts to convert raw EEG data into **spectrogram images**.
* Includes filtering (notch, high-pass), re-referencing, NaN handling, and spectrogram creation.
* Output: spectrograms used as input for Vision Transformer models.
* See [`README_ElectrodeWiseComparison.md`](./Preprocessing/SpectrogramCreation/README_Spectrogram.md).

### 2. Electrode-Wise Comparison

* Model finetuning scripts to evaluate **individual electrodes or subsets of electrodes**.
* Includes dataloaders, ViT model definitions, and training utilities.
* See [`README_ElectrodeWiseComparison.md`](./Electrode_Wise_Comparison/ModelFinetuning/README_ElectrodeWiseComparison.md).

### 3. Patient-Wise Comparison

* **Leave-One-Out (LOO) patient-wise finetuning** for model evaluation.
* Supports **multi-GPU training**.
* Post-hoc analysis with visualization (confusion matrices, threshold vs. accuracy, per-patient performance).
* See [`README_PatientWise.MD`](./Patient_Wise_Comparison/ModelFinetuning/README_PatientWise.MD) and [`README_analysis.md`](./Patient_Wise_Comparison/Analysis/README_analysis.md).

---

## Setup

Each submodule provides environment files:

* `requirements.txt` for Python dependencies.
* `GMMLenv_fixed.yml` for Conda environments.

Typical setup:

```bash
conda env create -f GMMLenv_fixed.yml
conda activate gmml_env
```

---

## Usage

1. **Preprocessing EEG data:**

   * Run MATLAB scripts in `Preprocessing/SpectrogramCreation/` to generate spectrogram images.

2. **Model Finetuning:**

   * Use scripts in `Electrode_Wise_Comparison/ModelFinetuning/` or
     `Patient_Wise_Comparison/ModelFinetuning/` depending on evaluation type.

3. **Analysis:**

   * Use `Patient_Wise_Comparison/Analysis/` for post-hoc analysis and visualization of results.

---

## Notes

* Subdirectories include dedicated README files with more detailed instructions.
* Results (plots, confusion matrices, etc.) are stored under `Analysis/Graph/`.
* The project is modular: preprocessing → model finetuning → analysis.
