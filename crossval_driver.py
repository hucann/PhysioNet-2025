# This script performs cross-validation for PTB-XL and SaMi-Trop datasets.
# PTB-XL is split using stratified folds (leave-one-fold-out), while SaMi-Trop uses random splits with different seeds.

# The script create folder data_train/ and data_test/ for training and testing data for each fold, and run the pipeline with subprocess.

import os
import shutil
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluate_model import evaluate_model

ptbxl_metadata = pd.read_csv('data/ptbxl_database.csv')
samitrop_dir = 'data/samitrop_output'
ptbxl_dir = 'data/ptbxl_output'
folds = list(range(1, 11))  # PTB-XL has 10 folds
all_scores = []

for fold in folds:
    print(f"\n=== Running Fold {fold} ===")

    # PTB-XL: stratified CV (leave-one-fold-out)
    df_test = ptbxl_metadata[ptbxl_metadata['strat_fold'] == fold]
    df_train = ptbxl_metadata[ptbxl_metadata['strat_fold'] != fold]

    # SaMi-Trop: random split with different seeds
    hea_files = [f for f in os.listdir(samitrop_dir) if f.endswith('.hea')]
    record_ids = sorted({os.path.splitext(f)[0] for f in hea_files})
    train_ids, temp_ids = train_test_split(record_ids, test_size=0.2, random_state=fold)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=fold)

    # Clear & recreate folders
    for path in ['data_train', 'data_test', 'outputs']:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    os.makedirs('data_train/ptbxl_output', exist_ok=True)
    os.makedirs('data_train/samitrop_output', exist_ok=True)
    os.makedirs('data_test/ptbxl_output', exist_ok=True)
    os.makedirs('data_test/samitrop_output', exist_ok=True)

    # Copy PTB-XL files
    def copy_ptbxl(df, root):
        for _, row in df.iterrows():
            rel_path = row['filename_hr'].replace('records500/', '')
            for ext in ['.hea', '.dat']:
                src = os.path.join(ptbxl_dir, rel_path + ext)
                dst = os.path.join(root, 'ptbxl_output', os.path.basename(src))
                if os.path.exists(src): shutil.copy2(src, dst)

    copy_ptbxl(df_train, 'data_train')
    copy_ptbxl(df_test, 'data_test')

    # Copy SaMi-Trop files
    def copy_samitrop(ids, root):
        for rid in ids:
            for ext in ['.hea', '.dat']:
                src = os.path.join(samitrop_dir, f"{rid}{ext}")
                dst = os.path.join(root, 'samitrop_output', f"{rid}{ext}")
                if os.path.exists(src): shutil.copy2(src, dst)

    copy_samitrop(train_ids, 'data_train')
    copy_samitrop(test_ids, 'data_test')  # you can also use val_ids here

    # Run training + inference
    subprocess.run(["python", "train_model.py", "-d", "data_train", "-m", "model", "-v"])
    subprocess.run(["python", "run_model.py", "-d", "data_test", "-m", "model", "-o", "outputs", "-v"])

    # Evaluate and store scores
    challenge_score, auroc, auprc, accuracy, f_measure = evaluate_model("data_test", "outputs")

    fold_result = {
        "fold": fold,
        "challenge_score": challenge_score,
        "AUROC": auroc,
        "AUPRC": auprc,
        "accuracy": accuracy,
        "F1": f_measure
    }
    all_scores.append(fold_result)

    print(fold_result)


# Save full per-fold results
df = pd.DataFrame(all_scores)
df.to_csv("cv_scores.csv", index=False)

# Compute and print summary
summary = df.describe().loc[["mean", "std"]]
summary.to_csv("cv_summary.txt")

print("\n=== CV Results Summary ===")
print(summary)