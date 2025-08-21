import os
import csv
from PIL import Image

# Define base directory
base_dir = "/media/enver/Seagate Portable Drive/Delerium-EEG/Spectrograms/ByElectrodes"

# Choose whether to delete original images
delete_original = True

# Valid formats to convert
valid_extensions = ['.jpg', '.jpeg', '.bmp', '.tiff']

# Label definitions
label_0_patients = {'JH', 'HC', 'WE', 'AM', 'LD'}
label_1_patients = {'LW', 'RT', 'NM', 'JL', 'AB'}

# --------------------------
# STEP 1: Convert to PNG
# --------------------------
def convert_all_to_png():
    for electrode in os.listdir(base_dir):
        electrode_path = os.path.join(base_dir, electrode)
        if not os.path.isdir(electrode_path):
            continue

        for subfolder in ['train', 'test']:
            folder_path = os.path.join(electrode_path, subfolder)
            if not os.path.isdir(folder_path):
                print(f"Skipping {electrode}/{subfolder} (folder not found)")
                continue

            for file in os.listdir(folder_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    name, _ = os.path.splitext(file)  # Preserve original casing
                    full_path = os.path.join(folder_path, file)
                    new_path = os.path.join(folder_path, name + ".png")

                    try:
                        with Image.open(full_path) as img:
                            img = img.convert("RGB")
                            img.save(new_path, "PNG")
                            print(f"Converted: {full_path} → {new_path}")

                            if delete_original:
                                os.remove(full_path)
                                print(f"Deleted original: {full_path}")
                    except Exception as e:
                        print(f"Failed to convert {full_path}: {e}")

# --------------------------
# STEP 2: Generate CSVs
# --------------------------
def generate_csvs():
    for electrode in os.listdir(base_dir):
        electrode_path = os.path.join(base_dir, electrode)
        if not os.path.isdir(electrode_path):
            continue

        for subfolder in ['train', 'test']:
            folder_path = os.path.join(electrode_path, subfolder)
            if not os.path.isdir(folder_path):
                continue

            entries = []

            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".png"):
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        patient_code = parts[1]
                        label = None
                        if patient_code in label_0_patients:
                            label = 0
                        elif patient_code in label_1_patients:
                            label = 1
                        else:
                            print(f"❗ Unknown patient code '{patient_code}' in {filename}")
                            continue

                        name_without_ext = os.path.splitext(filename)[0]
                        entries.append([name_without_ext, label])

            if entries:
                csv_name = "test_ds.csv" if subfolder == "test" else "train_ds.csv"
                csv_path = os.path.join(folder_path, csv_name)

                with open(csv_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(entries)

                print(f"📝 CSV written: {csv_path}")
            else:
                print(f"⚠️ No labeled images found in {electrode}/{subfolder}")

# --------------------------
# Run both steps
# --------------------------
if __name__ == "__main__":
    convert_all_to_png()
    generate_csvs()