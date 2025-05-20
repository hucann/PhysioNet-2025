#!/usr/bin/env python3

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    base_dir = 'data'
    code15_dir = os.path.join(base_dir, 'code15_output')
    selected_list_path = 'selected_records.txt'
    exam_csv_path = os.path.join(base_dir, 'exams.csv')

    # ✅ CORRECT folder destinations beside other datasets
    train_dir = os.path.join(base_dir, 'data_train', 'code15_output')
    test_dir = os.path.join(base_dir, 'data_test', 'code15_output')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Load selected records
    with open(selected_list_path, 'r') as f:
        selected_records = [line.strip() for line in f if line.strip()]

    # Load exam.csv
    df = pd.read_csv(exam_csv_path)

    # Map record path (e.g., exams_part1/113) to patient_id
    record_to_patient = {}
    for record in selected_records:
        folder, exam_id_str = record.split('/')
        exam_id = int(os.path.splitext(exam_id_str)[0])
        patient_row = df[df['exam_id'] == exam_id]
        if not patient_row.empty:
            record_to_patient[record] = patient_row.iloc[0]['patient_id']

    # Convert to DataFrame
    df_records = pd.DataFrame({
        'record': list(record_to_patient.keys()),
        'patient_id': list(record_to_patient.values())
    })

    # Patient-level split
    unique_patients = df_records['patient_id'].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

    df_records['split'] = df_records['patient_id'].apply(lambda pid: 'train' if pid in train_patients else 'test')

    # Copy files
    for _, row in df_records.iterrows():
        record = row['record']
        split = row['split']
        src_folder = os.path.join(code15_dir, os.path.dirname(record))
        file_prefix = os.path.basename(record)
        src_hea = os.path.join(src_folder, file_prefix + '.hea')
        src_dat = os.path.join(src_folder, file_prefix + '.dat')
        dst_dir = train_dir if split == 'train' else test_dir

        if os.path.exists(src_hea) and os.path.exists(src_dat):
            shutil.copy2(src_hea, dst_dir)
            shutil.copy2(src_dat, dst_dir)
        else:
            print(f"⚠️ Missing files for: {record}")

    print("✅ Split complete.")
    print(f"Train: {sum(df_records['split'] == 'train')} records")
    print(f"Test: {sum(df_records['split'] == 'test')} records")

if __name__ == '__main__':
    main()