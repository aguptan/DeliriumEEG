import shutil
from PIL import Image
import csv
import os

# === User-defined parameters ===
time_duration = "15min"  

source_dir = f"/media/enver/Seagate Portable Drive/Delerium-EEG/Spectrograms2/{time_duration}/ByPatient"
destination_dir = f"/media/enver/Seagate Portable Drive/Delerium-EEG/Spectrograms2/{time_duration}/ByElectrode"

# === STEP 1: Reorganize by electrode ===
os.makedirs(destination_dir, exist_ok=True)

for patient_folder in os.listdir(source_dir):
    patient_path = os.path.join(source_dir, patient_folder)

    if os.path.isdir(patient_path):
        for electrode_folder in os.listdir(patient_path):
            electrode_path = os.path.join(patient_path, electrode_folder)

            if os.path.isdir(electrode_path):
                new_name = f"{patient_folder}_{electrode_folder}"
                electrode_dest_folder = os.path.join(destination_dir, electrode_folder)
                os.makedirs(electrode_dest_folder, exist_ok=True)
                new_dest_path = os.path.join(electrode_dest_folder, new_name)

                shutil.copytree(electrode_path, new_dest_path, dirs_exist_ok=True)
                print(f"Copied: {electrode_path} -> {new_dest_path}")

print(" Step 1: Reorganized by electrode.")

# === STEP 2: Sort into 'train'/'test' ===
patient_groups = {
    "JH": 1, "HC": 2, "WE": 3, "AM": 4, "LD": 5,  # Train
    "LW": 6, "RT": 7, "NM": 8, "JL": 9, "AB": 10  # Test
}

for electrode_folder in os.listdir(destination_dir):
    electrode_path = os.path.join(destination_dir, electrode_folder)

    if os.path.isdir(electrode_path):
        train_path = os.path.join(electrode_path, "train")
        test_path = os.path.join(electrode_path, "test")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for patient_folder in os.listdir(electrode_path):
            patient_folder_path = os.path.join(electrode_path, patient_folder)

            if not os.path.isdir(patient_folder_path) or patient_folder in ["train", "test"]:
                continue

            patient_id = patient_folder[:2]
            patient_number = patient_groups.get(patient_id)

            if patient_number is not None:
                dest_path = train_path if patient_number <= 5 else test_path
                shutil.move(patient_folder_path, dest_path)
                print(f"Moved: {patient_folder_path} -> {dest_path}")

print(" Step 2: Sorted into 'train'/'test'.")

# === STEP 3: Flatten image directories ===
def flatten_folders(main_folder):
    for electrode in os.listdir(main_folder):
        electrode_path = os.path.join(main_folder, electrode)
        if os.path.isdir(electrode_path):
            for session in ['train', 'test']:
                session_path = os.path.join(electrode_path, session)
                if os.path.exists(session_path):
                    for root, _, files in os.walk(session_path):
                        for file in files:

                            if file.lower().endswith(('.jpg', '.jpeg')):
                                source = os.path.join(root, file)
                                dest = os.path.join(session_path, file)
                                base, ext = os.path.splitext(file)
                                counter = 1
                                while os.path.exists(dest):
                                    dest = os.path.join(session_path, f"{base}_{counter}{ext}")
                                    counter += 1
                                shutil.move(source, dest)
                                print(f"Moved: {source} → {dest}")
                    for root, dirs, files in os.walk(session_path, topdown=False):
                        if root != session_path and not os.listdir(root):
                            os.rmdir(root)
                            print(f"Removed: {root}")

flatten_folders(destination_dir)

print(" Step 3: Flattened directories.")

# === STEP 4: Convert images to PNG & Generate CSV ===
valid_extensions = ['.jpg', '.jpeg', '.bmp', '.tiff']
label_0_patients = {'JH', 'HC', 'WE', 'AM', 'LD'}
label_1_patients = {'LW', 'RT', 'NM', 'JL', 'AB'}

def convert_and_csv(base_dir, delete_original=True):
    for electrode in os.listdir(base_dir):
        electrode_path = os.path.join(base_dir, electrode)
        if not os.path.isdir(electrode_path): continue

        for session in ['train', 'test']:
            session_path = os.path.join(electrode_path, session)
            if not os.path.isdir(session_path): continue

            entries = []
            for file in os.listdir(session_path):
                ext = os.path.splitext(file)[1].lower()
                filepath = os.path.join(session_path, file)

                if ext in valid_extensions:
                    png_file = os.path.splitext(file)[0] + ".png"
                    png_path = os.path.join(session_path, png_file)
                    with Image.open(filepath) as img:
                        img.convert("RGB").save(png_path, "PNG")
                        if delete_original: os.remove(filepath)
                        print(f"Converted: {filepath} → {png_path}")

                if file.lower().endswith(".png"):
                    parts = file.split("_")
                    if len(parts) >= 2:
                        patient_code = parts[1]
                        label = 0 if patient_code in label_0_patients else 1 if patient_code in label_1_patients else None
                        if label is not None:
                            entries.append([os.path.splitext(file)[0], label])

            csv_name = f"{session}_ds.csv"
            csv_path = os.path.join(session_path, csv_name)
            if entries:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(entries)
                print(f" CSV generated: {csv_path}")

convert_and_csv(destination_dir)

print(" Step 4: Images converted & CSV files generated.")