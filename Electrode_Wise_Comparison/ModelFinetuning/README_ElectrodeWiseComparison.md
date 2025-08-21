# Electrode-Wise Vision Transformer Fine-Tuning for EEG/ECoG

This repository extends the original **VisionTransformerModel** codebase to enable systematic **electrode-wise comparison** in EEG/ECoG classification tasks.

The modified workflow introduces a looping mechanism that fine-tunes a Vision Transformer model separately for each electrode and aggregates the results for analysis.

---

## Key Differences from the Original Code

* **Original Code (VisionTransformerModel)**

  * Implements electrode-wise fine-tuning of a Vision Transformer on EEG/ECoG spectrograms.
  * Includes model definitions (`vision_transformer.py`), dataset loaders (`ECOG90S_dataloader.py`), and training utilities (`utils.py`).

* **Code (`loop_Model_Finetuning.py`)**

  * Automates fine-tuning across individual electrodes.
  * Trains and evaluates a model for each electrode in sequence.
  * Collects per-electrode performance metrics for direct comparison.
  * Preserves compatibility with the original training pipeline.

---

## Repository Structure

```
ModelFinetuning/
├── VisionTransformerModel/
│   ├── finetune_EEG.py              # Original fine-tuning script
│   ├── vision_transformer.py        # Vision Transformer model definitions
│   ├── ECOG90S_dataloader.py        # EEG/ECoG dataset loader
│   ├── utils.py                     # Training utilities
│
├── loop_Model_Finetuning.py         # Modified script looping over electrodes
```

---

## Installation

Clone the repository and install dependencies:

```
git clone <your-fork-url>
cd ModelFinetuning
pip install -r requirements.txt
```

Dependencies include PyTorch, timm (for Vision Transformers), and standard scientific libraries (numpy, pandas, scipy, scikit-learn).

---

## Usage

### Run electrode-wise fine-tuning loop

```
python loop_Model_Finetuning.py --config configs/default.yaml
```

The loop script will:

* Iterate over all electrodes in the dataset
* Train and evaluate a model per electrode
* Save metrics and logs for electrode-level comparison

---

## Outputs

* Per-electrode metrics (accuracy, AUC, loss, etc.)
* Logs and model checkpoints for each electrode
* Comparison tables summarizing electrode-wise performance

---

## Acknowledgments

This work builds on the original **VisionTransformerModel** codebase for patient-wise EEG/ECoG fine-tuning. The modifications extend it to support systematic electrode-wise evaluation for channel-level analysis and interpretability.
