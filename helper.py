import os
import random
import shutil

# Define paths
train_dir = 'data_train/samitrop_output'
test_dir = 'data_test/samitrop_output'

# Create destination directory if not exists
os.makedirs(test_dir, exist_ok=True)

# List all .dat files (we assume .hea exists with same name)
dat_files = [f for f in os.listdir(train_dir) if f.endswith('.dat')]
record_ids = [os.path.splitext(f)[0] for f in dat_files]

# Shuffle and select 20%
random.shuffle(record_ids)
n_move = int(len(record_ids) * 0.2)
to_move = record_ids[:n_move]

# Move each .dat and corresponding .hea file
for rid in to_move:
    for ext in ['.dat', '.hea']:
        src = os.path.join(train_dir, rid + ext)
        dst = os.path.join(test_dir, rid + ext)
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"Warning: {src} not found")

print(f"Moved {n_move} record(s) to {test_dir}")