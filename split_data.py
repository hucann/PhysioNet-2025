# This script splits the PTB-XL and SaMi-Trop datasets into training, validation, and test sets. (80% train, 10% val, 10% test)
# PTB-XL is split with stratified folds to prevent patient-level data leakage, while SaMi-Trop is split randomly.

# data_train_val/
# ├── train/
# │   ├── ptbxl_output/
# │   └── samitrop_output/
# └── val/
#     ├── ptbxl_output/
#     └── samitrop_output/
# data_test/
# ├── ptbxl_output/
# └── samitrop_output/

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# === Paths ===
base_dir = 'data'
ptbxl_dir = os.path.join(base_dir, 'ptbxl_output')
samitrop_dir = os.path.join(base_dir, 'samitrop_output')
ptbxl_metadata_path = os.path.join(base_dir, 'ptbxl_database.csv')

train_val_base = 'data_train_val'
test_base = 'data_test'

# === Create Directory Structure ===
for split in ['train', 'val']:
    os.makedirs(os.path.join(train_val_base, split, 'ptbxl_output'), exist_ok=True)
    os.makedirs(os.path.join(train_val_base, split, 'samitrop_output'), exist_ok=True)

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

def copy_ptbxl_files(df, dest_root):
    for _, row in df.iterrows():
        rel_path = row['filename_hr'].replace('records500/', '') 
        src_hea = os.path.join(ptbxl_dir, rel_path + '.hea')
        src_dat = os.path.join(ptbxl_dir, rel_path + '.dat')
        dst_dir = os.path.join(dest_root, 'ptbxl_output')
        if os.path.exists(src_hea) and os.path.exists(src_dat):
            shutil.copy2(src_hea, dst_dir)
            shutil.copy2(src_dat, dst_dir)
        else:
            print(f"[PTB-XL] Missing file: {rel_path}")

copy_ptbxl_files(df_train, os.path.join(train_val_base, 'train'))
copy_ptbxl_files(df_val, os.path.join(train_val_base, 'val'))
copy_ptbxl_files(df_test, test_base)

# === SaMi-Trop Split ===
hea_files = [f for f in os.listdir(samitrop_dir) if f.endswith('.hea')]
record_ids = sorted({os.path.splitext(f)[0] for f in hea_files})

train_ids, temp_ids = train_test_split(record_ids, test_size=0.2, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

def copy_samitrop_files(ids, dest_root):
    for rid in ids:
        hea_path = os.path.join(samitrop_dir, f"{rid}.hea")
        dat_path = os.path.join(samitrop_dir, f"{rid}.dat")
        dst_dir = os.path.join(dest_root, 'samitrop_output')
        if os.path.exists(hea_path) and os.path.exists(dat_path):
            shutil.copy2(hea_path, dst_dir)
            shutil.copy2(dat_path, dst_dir)
        else:
            print(f"[SaMi-Trop] Missing file: {rid}")

copy_samitrop_files(train_ids, os.path.join(train_val_base, 'train'))
copy_samitrop_files(val_ids, os.path.join(train_val_base, 'val'))
copy_samitrop_files(test_ids, test_base)

# === Optional: Save record lists for tracking ===
# pd.Series(train_ids).to_csv("train_samitrop_ids.csv", index=False)
# pd.Series(val_ids).to_csv("val_samitrop_ids.csv", index=False)
# pd.Series(test_ids).to_csv("test_samitrop_ids.csv", index=False)




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