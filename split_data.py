# This script splits the PTB-XL and SaMi-Trop datasets into training, validation, and test sets. (80% train, 10% val, 10% test)
# PTB-XL is split with stratified folds to prevent patient-level data leakage, while SaMi-Trop is split randomly with fixed seed.

# data_train/
# ├── train/
# │   ├── ptbxl_output/
# │   └── samitrop_output/
# └── val/
#     ├── ptbxl_output/
#     └── samitrop_output/
############################ OR
# data_train/
# ├── ptbxl_output/
# └── samitrop_output/
############################
# data_test/
# ├── ptbxl_output/
# └── samitrop_output/

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
split_train_val = False  # ✅ Set True for separate train/val, False to merge

# === Paths ===
base_dir = 'data'
ptbxl_dir = os.path.join(base_dir, 'ptbxl_output')
samitrop_dir = os.path.join(base_dir, 'samitrop_output')
ptbxl_metadata_path = os.path.join(base_dir, 'ptbxl_database.csv')

train_base = os.path.join(base_dir, 'data_train')
test_base = os.path.join(base_dir, 'data_test')

# Create folders
if split_train_val:
    for split in ['train', 'val']:
        os.makedirs(os.path.join(train_base, split, 'ptbxl_output'), exist_ok=True)
        os.makedirs(os.path.join(train_base, split, 'samitrop_output'), exist_ok=True)
else:
    os.makedirs(os.path.join(train_base, 'ptbxl_output'), exist_ok=True)
    os.makedirs(os.path.join(train_base, 'samitrop_output'), exist_ok=True)

os.makedirs(os.path.join(test_base, 'ptbxl_output'), exist_ok=True)
os.makedirs(os.path.join(test_base, 'samitrop_output'), exist_ok=True)

# === PTB-XL Split ===
ptbxl_df = pd.read_csv(ptbxl_metadata_path)
train_folds = list(range(1, 9))
val_fold = 9
test_fold = 10

df_train = ptbxl_df[ptbxl_df['strat_fold'].isin(train_folds)]
df_val = ptbxl_df[ptbxl_df['strat_fold'] == val_fold]
df_test = ptbxl_df[ptbxl_df['strat_fold'] == test_fold]

def copy_ptbxl_files(df, output_dir):
    for _, row in df.iterrows():
        rel_path = row['filename_hr'].replace('records500/', '')
        src_hea = os.path.join(ptbxl_dir, rel_path + '.hea')
        src_dat = os.path.join(ptbxl_dir, rel_path + '.dat')
        dst_dir = os.path.join(output_dir, 'ptbxl_output')
        if os.path.exists(src_hea) and os.path.exists(src_dat):
            shutil.copy2(src_hea, dst_dir)
            shutil.copy2(src_dat, dst_dir)
        else:
            print(f"[PTB-XL] Missing file: {rel_path}")

# Copy PTB-XL
if split_train_val:
    copy_ptbxl_files(df_train, os.path.join(train_base, 'train'))
    copy_ptbxl_files(df_val, os.path.join(train_base, 'val'))
else:
    combined_df = pd.concat([df_train, df_val])
    copy_ptbxl_files(combined_df, train_base)

copy_ptbxl_files(df_test, test_base)

# === SaMi-Trop Split ===
hea_files = [f for f in os.listdir(samitrop_dir) if f.endswith('.hea')]
record_ids = sorted({os.path.splitext(f)[0] for f in hea_files})

train_ids, temp_ids = train_test_split(record_ids, test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

def copy_samitrop_files(ids, output_dir):
    for rid in ids:
        hea_path = os.path.join(samitrop_dir, f"{rid}.hea")
        dat_path = os.path.join(samitrop_dir, f"{rid}.dat")
        dst_dir = os.path.join(output_dir, 'samitrop_output')
        if os.path.exists(hea_path) and os.path.exists(dat_path):
            shutil.copy2(hea_path, dst_dir)
            shutil.copy2(dat_path, dst_dir)
        else:
            print(f"[SaMi-Trop] Missing file: {rid}")

# Copy SaMi-Trop
if split_train_val:
    copy_samitrop_files(train_ids, os.path.join(train_base, 'train'))
    copy_samitrop_files(val_ids, os.path.join(train_base, 'val'))
else:
    combined_ids = train_ids + val_ids
    copy_samitrop_files(combined_ids, train_base)

copy_samitrop_files(test_ids, test_base)

print("\n✅ Dataset split complete.")
print(f"Saved in: {train_base}/ and {test_base}/")



# =======================================================================
# import os
# import random
# import shutil

# # Define paths
# train_dir = 'data_train/samitrop_output'
# test_dir = 'data_test/samitrop_output'

# # Create destination directory if not exists
# os.makedirs(test_dir, exist_ok=True)

# # List all .dat files (we assume .hea exists with same name)
# dat_files = [f for f in os.listdir(train_dir) if f.endswith('.dat')]
# record_ids = [os.path.splitext(f)[0] for f in dat_files]

# # Shuffle and select 20%
# random.shuffle(record_ids)
# n_move = int(len(record_ids) * 0.2)
# to_move = record_ids[:n_move]

# # Move each .dat and corresponding .hea file
# for rid in to_move:
#     for ext in ['.dat', '.hea']:
#         src = os.path.join(train_dir, rid + ext)
#         dst = os.path.join(test_dir, rid + ext)
#         if os.path.exists(src):
#             shutil.move(src, dst)
#         else:
#             print(f"Warning: {src} not found")

# print(f"Moved {n_move} record(s) to {test_dir}")


# =======================================================================
# Check prevalence of positive cases in the dataset
# import os
# from helper_code import load_label, find_records

# def check_prevalence(data_folder, verbose=True):
#     # Get list of all records (e.g., exams_part0/113)
#     records = find_records(data_folder)
#     num_records = len(records)

#     if num_records == 0:
#         raise FileNotFoundError('No records found in the data folder.')

#     positive = 0
#     for i, rec in enumerate(records):
#         if verbose and i % 1000 == 0:
#             print(f"Processing record {i+1}/{num_records}...")

#         record_path = os.path.join(data_folder, rec)
#         label = load_label(record_path)
#         if label:
#             positive += 1

#     prevalence = positive / num_records
#     print("\n====== Prevalence Statistics ======")
#     print(f"Total records:     {num_records}")
#     print(f"Positive cases:    {positive}")
#     print(f"Prevalence rate:   {prevalence:.4f} ({prevalence*100:.2f}%)")

# if __name__ == '__main__':
#     data_folder = 'data/code15_output/exams_part1'  # Update this if needed
#     check_prevalence(data_folder)