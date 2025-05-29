import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "data"
SRC_FOLDER = os.path.join(DATA_DIR, "code15_output")
TRAIN_FOLDER = os.path.join("data_train", "code15_output")
TEST_FOLDER = os.path.join("data_test", "code15_output")
META_CSV = "data/code15_meta.csv"

# Ensure output directories exist
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# Load metadata
df = pd.read_csv(META_CSV)

# Filter to only exams from exams_part0.hdf5
df = df[df['trace_file'] == 'exams_part0.hdf5']

# Train-test split by patient_id
unique_patients = df['patient_id'].unique()
train_patients, test_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=42
)

# Assign split label
df['split'] = df['patient_id'].apply(
    lambda pid: 'train' if pid in train_patients else 'test'
)

# Copy files
for _, row in df.iterrows():
    exam_id = str(row['exam_id'])
    split = row['split']

    src_dat = os.path.join(SRC_FOLDER, f"{exam_id}.dat")
    src_hea = os.path.join(SRC_FOLDER, f"{exam_id}.hea")

    if not os.path.exists(src_dat) or not os.path.exists(src_hea):
        print(f"⚠️ Missing file(s) for exam_id {exam_id}")
        continue

    dst_folder = TRAIN_FOLDER if split == 'train' else TEST_FOLDER
    shutil.copy2(src_dat, dst_folder)
    shutil.copy2(src_hea, dst_folder)
    # print(f"✅ Copied {exam_id} to {split}")

print("✅ Train-test split completed.")