import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unified_container import UnifiedResultsContainer, UnifiedAnalysisHelper
from scipy.stats import linregress
import matplotlib.patches as mpatches

# === Load container ===
container_path = r"C:\Users\agupt\Desktop\Shoykhet_Lab\DeliriumEEG\Analysis\LeaveOneOut\8.4.25_LOO\unified_container_8.24.pkl"
with open(container_path, "rb") as f:
    container = pickle.load(f)

helper = UnifiedAnalysisHelper(container)
time_windows = list(container.results_by_time_window.keys())
EPOCH_RANGE = (25, 35)
color_map = {0: "blue", 1: "red"}

# === Collect patient-level data with adjusted accuracy and correctness ===
patient_data_by_window = {}
for time_window in time_windows:
    df = helper.patient_df(time_window=time_window, epoch_range=EPOCH_RANGE)
    df["adjusted_accuracy"] = df.apply(
        lambda row: (row["positive_images"] / row["total_images"]) if row["true_label"] == 1
        else (1 - row["positive_images"] / row["total_images"]),
        axis=1
    )
    df["correct_images"] = df.apply(
        lambda row: row["positive_images"] if row["true_label"] == 1
        else row["total_images"] - row["positive_images"],
        axis=1
    )
    patient_data_by_window[time_window] = df

# === Combine across all time windows ===
all_df = pd.concat(patient_data_by_window.values(), ignore_index=True)

# === === FIGURE 1: Heatmap of Adjusted Accuracy at Best Epoch === ===
best_df = all_df.loc[all_df.groupby(["fold_id", "test_set"])["adjusted_accuracy"].idxmax()]
heatmap_df = best_df.pivot(index="fold_id", columns="test_set", values="adjusted_accuracy")
fold_id_to_class = best_df.drop_duplicates("fold_id").set_index("fold_id")["true_label"].to_dict()
y_labels = [f"{pid} (Class {fold_id_to_class.get(pid, '?')})" for pid in heatmap_df.index]

fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(heatmap_df.values, aspect='auto', cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(heatmap_df.columns)))
ax.set_xticklabels(heatmap_df.columns, rotation=45, ha='right')
ax.set_yticks(np.arange(len(heatmap_df.index)))
ax.set_yticklabels(y_labels)
ax.set_xlabel("Test Set")
ax.set_ylabel("Patient ID (with True Class)")
ax.set_title("Figure 1: Adjusted Accuracy Heatmap (Best Epochs 25–35)")
fig.colorbar(cax, ax=ax, label='Adjusted Accuracy')
plt.tight_layout()
plt.show()

# === === FIGURE 2: Per-Patient Weighted Accuracy by Test Set === ===
agg_df = (
    all_df.groupby(["fold_id", "true_label", "test_set"])
    .agg(correct_images=("correct_images", "sum"), total_images=("total_images", "sum"))
    .reset_index()
)
agg_df["weighted_accuracy"] = agg_df["correct_images"] / agg_df["total_images"]

fig, ax = plt.subplots(figsize=(10, 6))
test_sets = sorted(agg_df["test_set"].unique())
positions = range(len(test_sets))
width = 0.6

for i, test_set in enumerate(test_sets):
    subset = agg_df[agg_df["test_set"] == test_set]
    jitter = (np.random.rand(len(subset)) - 0.5) * width * 0.4
    for (x_jit, y_val, label) in zip(jitter, subset["weighted_accuracy"], subset["true_label"]):
        ax.plot(positions[i] + x_jit, y_val, 'o', color=color_map[label], alpha=0.8)
    mean_acc = subset["weighted_accuracy"].mean()
    ax.hlines(mean_acc, positions[i] - width / 2, positions[i] + width / 2, colors="black", linestyles="--")

ax.set_xticks(positions)
ax.set_xticklabels(test_sets, rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.set_ylabel("Image-Weighted Adjusted Accuracy (Epochs 25–35)")
ax.set_xlabel("Test Set")
ax.set_title("Figure 2: Per-Patient Weighted Accuracy by Test Set")
ax.grid(True, linestyle='--', alpha=0.5)
legend_handles = [
    mpatches.Patch(color='red', label='Class 1'),
    mpatches.Patch(color='blue', label='Class 0'),
    mpatches.Patch(color='black', label='Mean Line')
]
ax.legend(handles=legend_handles)
plt.tight_layout()
plt.show()

# === === FIGURE 3: Optimal Threshold vs Adjusted Accuracy === ===
thresholds_df = best_df[["fold_id", "test_set", "optimal_threshold"]]
merged_df = pd.merge(agg_df, thresholds_df, on=["fold_id", "test_set"], how="inner")

fig, ax = plt.subplots(figsize=(10, 6))
for label in [0, 1]:
    subset = merged_df[merged_df["true_label"] == label]
    ax.scatter(
        subset["optimal_threshold"],
        subset["weighted_accuracy"],
        label=f"Class {label}",
        color=color_map[label],
        alpha=0.7
    )

slope, intercept, r, p, _ = linregress(
    merged_df["optimal_threshold"],
    merged_df["weighted_accuracy"]
)


ax.set_xlabel("Optimal Threshold")
ax.set_ylabel("Weighted Adjusted Accuracy (Epochs 25–35)")
ax.set_title("Figure 3: Optimal Threshold vs Adjusted Accuracy")
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# === Parameters ===
EPOCH_RANGE = (25, 35)
normalize = False  # Set True to normalize (percentages)
time_windows = list(container.results_by_time_window.keys())

# === Loop through time windows ===
for time_window in time_windows:
    df = helper.patient_df(time_window=time_window, epoch_range=EPOCH_RANGE)
    test_sets = df["test_set"].unique()

    # Prepare figure layout
    n_cols = 2
    n_rows = int(np.ceil(len(test_sets) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axs = axs.flatten()

    # Compute max value for shared scale
    cm_max = 0
    confusion_matrices = {}

    for test_set in test_sets:
        subset = df[df["test_set"] == test_set]
        cm = confusion_matrix(subset["true_label"], subset["pred_label"], labels=[0, 1])
        confusion_matrices[test_set] = cm
        cm_max = max(cm_max, cm.max())

    # Plot confusion matrices
    for i, test_set in enumerate(test_sets):
        ax = axs[i]
        cm = confusion_matrices[test_set]

        if normalize:
            cm_sum = cm.sum()
            cm_display = cm / cm_sum if cm_sum > 0 else np.zeros_like(cm)
        else:
            cm_display = cm

        im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=(1.0 if normalize else cm_max))

        for (x, y), val in np.ndenumerate(cm_display):
            label = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(y, x, label, ha="center", va="center", fontsize=12, color="black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.text(
            1.05, 0.5,  # x=right outside, y=center
            f"{test_set}",
            transform=ax.transAxes,
            fontsize=12,
            va='center',
            ha='left',
            rotation=0
        )


    # Remove unused axes
    for j in range(len(test_sets), len(axs)):
        fig.delaxes(axs[j])

    # Final layout
    fig.suptitle(f"Figure 4: Confusion Matrices by Test Set\nTime Window: {time_window} (Epochs {EPOCH_RANGE[0]}–{EPOCH_RANGE[1]})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unified_container import UnifiedResultsContainer, UnifiedAnalysisHelper

# === Load container ===
container_path = r"C:\Users\agupt\Desktop\Shoykhet_Lab\DeliriumEEG\Analysis\LeaveOneOut\8.4.25_LOO\unified_container_8.24.pkl"
with open(container_path, "rb") as f:
    container = pickle.load(f)

helper = UnifiedAnalysisHelper(container)
time_windows = list(container.results_by_time_window.keys())
EPOCH_RANGE = (25, 35)

# === Get patient-level data and classify by TP / FP / FN / TN ===
patient_data = []
for time_window in time_windows:
    df = helper.patient_df(time_window=time_window, epoch_range=EPOCH_RANGE)
    df = df[df["epoch"].between(*EPOCH_RANGE)]
    df["correct"] = df["true_label"] == df["pred_label"]
    df["error_type"] = df.apply(
        lambda row: "TP" if row["true_label"] == 1 and row["pred_label"] == 1 else
                    "TN" if row["true_label"] == 0 and row["pred_label"] == 0 else
                    "FN" if row["true_label"] == 1 and row["pred_label"] == 0 else
                    "FP", axis=1
    )
    patient_data.append(df)

combined_df = pd.concat(patient_data, ignore_index=True)

# === Collapse to best epoch per patient × test_set (same as earlier logic) ===
best_epoch_df = combined_df.loc[
    combined_df.groupby(["fold_id", "test_set"])["adjusted_accuracy"].idxmax()
].copy()

# === Count TP / FP / FN / TN per test set ===
confusion_counts = (
    best_epoch_df.groupby(["test_set", "error_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# === Plot grouped bar plot ===
error_types = ["TP", "TN", "FP", "FN"]
colors = {"TP": "green", "TN": "blue", "FP": "orange", "FN": "red"}

x = np.arange(len(confusion_counts["test_set"]))
bar_width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

for i, error_type in enumerate(error_types):
    values = confusion_counts[error_type] if error_type in confusion_counts else [0] * len(x)
    ax.bar(x + i * bar_width, values, width=bar_width, label=error_type, color=colors[error_type])

# === Final touches ===
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(confusion_counts["test_set"], rotation=45, ha='right')
ax.set_ylabel("Count")
ax.set_title("Figure 5: TP / TN / FP / FN Counts per Test Set (Best Epochs 25–35)")
ax.legend(title="Error Type")
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
