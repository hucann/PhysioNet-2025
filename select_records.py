#!/usr/bin/env python3

# Select records based on labels and model outputs, from exams_part0 and exams_part1.
# 1) label == 1
# 2) model output probability < 0.01 or > 0.99
# Total number of selected records: 819 out of aroud 40,000 records.

import os
from helper_code import load_label

def select_records(data_folder, output_folder, save_path='selected_records.txt', verbose=True):
    selected = []

    for root, _, files in os.walk(data_folder):
        for file in files:
            if not file.endswith('.hea'):
                continue

            # Full and relative path to .hea
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, data_folder)  # e.g. exams_part1/14.hea
            record = os.path.splitext(rel_path)[0]  # -> exams_part1/14

            if verbose:
                print(f"Processing: {record}")

            # Load label
            record_path = os.path.join(data_folder, record)
            try:
                label = load_label(record_path)
            except:
                if verbose:
                    print(f"  ⚠️  Could not load label for {record}")
                continue

            # Load model output
            output_file = os.path.join(output_folder, record + '.txt')
            if not os.path.exists(output_file):
                if verbose:
                    print(f"  ⚠️  Missing output file for {record}")
                continue

            try:
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                prob_line = next((l for l in lines if 'Chagas probability:' in l), None)
                prob = float(prob_line.split(':')[-1].strip()) if prob_line else None
            except:
                if verbose:
                    print(f"  ⚠️  Could not parse output for {record}")
                continue

            # # Apply selection criteria
            # if label == 1 or (prob is not None and (prob < 0.01 or prob > 0.99)):
            #     selected.append(record)
            # Apply selection criteria
            if prob is not None and (prob < 0.01 or prob > 0.99):
                selected.append(record)

    # Save to text file
    with open(save_path, 'w') as f:
        for r in selected:
            f.write(f"{r}\n")

    print(f"\n✅ Total selected: {len(selected)}")
    print(f"📄 Saved to: {save_path}")

if __name__ == '__main__':
    DATA_FOLDER = 'data/code15_output'
    OUTPUT_FOLDER = 'outputs'
    select_records(DATA_FOLDER, OUTPUT_FOLDER)