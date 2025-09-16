## Why I changed the values for the Spectrogram Creation


## 1) Longer analysis window (8 s → 1600 samples) and higher overlap (75%)

* **What changed:** `window_length` increased from 200 to 1600 samples (8 s at 200 Hz), and `overlap_length` set to 1200 (75%).
* **Why:** True frequency resolution is `fs / window_length`. Moving from 1 s (1.0 Hz resolution) to 8 s gives **0.125 Hz** resolution. That’s critical to (a) cleanly separate **theta (4–7 Hz)** from **alpha (8–13 Hz)** and (b) detect **alpha slowing**—both central to delirium signatures.
* **Side effect:** With 75% overlap, your time hop is 2 s, which lowers variance in the averaged spectrogram without losing too much temporal responsiveness.

## 2) Compute only the band of interest via a frequency vector (0.5–45 Hz)

* **What changed:** Instead of a scalar `nfft`, we pass a **frequency vector** `f_vec = f_low:df:f_high` to `spectrogram` (here, 0.5:0.125:45).
* **Why:** MATLAB then computes **only** 0.5–45 Hz. This improves SNR (drops line noise/EMG-heavy high frequencies), reduces compute/memory, and makes the output directly aligned with delirium-relevant bands.

## 3) Explicitly use a **Hann** window

* **What changed:** We build `win = hann(window_length,'periodic')` and pass that window to `spectrogram` instead of passing a scalar length (which defaults to **Hamming**).
* **Why:** **Hann** has lower sidelobe levels and less spectral leakage into neighboring bins than Hamming for this use case, giving cleaner band boundaries and more trustworthy power in and near alpha/theta.

## 4) Convert to **power in dB** (not magnitude in dB)

* **What changed:** We now square the complex STFT magnitude first (`|S|²`) and then compute `10*log10(power + eps)`.
* **Why:** Bandpower, band ratios, spectral edge measures, and most EEG spectral quantities are defined in terms of **power** (or PSD). Using magnitude underestimates dynamic range and skews ratios; power in dB is physically meaningful and consistent with the literature.

## 5) (Recommended) Fixed dynamic range for all images (global `caxis`)

* **Current state:** Still using per-image `caxis([min, max])`, which **autoscale** the dynamic range.
* **Why change later:** A fixed range (e.g., learned once from training-set percentiles) makes identical power map to identical brightness/color everywhere, which stabilizes the ViT’s input distribution and improves generalization.
* **If you defer calibration:** At least pick a single fixed pair (e.g., **\[-100, -20] dB**) and keep it constant across train/val/test.

## 6) Colormap choice (parula or grayscale)

* **What changed:** Switched to `parula`.
* **Why:** `parula` (or viridis/magma) is more **perceptually uniform** and avoids the false edges that `jet` introduces. For ViT, **3×grayscale** (by replicating a single channel) is often a strong baseline too. Whatever you choose, keep it **consistent**.

## 7) Richer metadata for reproducibility

* **What changed:** Added fields for `freq_low`, `freq_high`, `df`, `num_freq_bins`, `window_type='hann'`, and `is_db_power=true`; kept `_nfft = NaN` for backward compatibility and flagged `_used_freq_vector=true`.
* **Why:** Downstream analysis and future audits can unambiguously reconstruct how each spectrogram was produced (window type/length, band limits, resolution, and scaling convention).

---

### What stays the same

* Your segmentation into 15/30/60/120-min blocks, directory structure, and 224×224 export all remain unchanged, so existing pipelines and filenames continue to work.

### Expected impact

* **Better band separation** (θ vs α) and **finer alpha-peak tracking** thanks to 0.125 Hz resolution.
* **Cleaner inputs** to the ViT by computing only 0.5–45 Hz.
* **More faithful power features** and ratios after switching to **power dB**.
* **Improved generalization** once you move from per-image autoscale to a **fixed global `caxis`**.
