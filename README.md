## ECG Signal Processing & Classification

This repository contains code and workflows for processing ECG signals and training models to classify Chagas Disease from multiple datasets (PTB-XL, SaMi-Trop, CODE15). 

---

### How to Run

1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Download the preprocessed datasets, place them in the `data/` directory as follows:
```
data/
├── code15_output/
├── ptbxl_output/
├── samitrop_output/
├── code15_meta.csv
└── ptbxl_meta.csv
```

3. Create training and testing dataset by running appropriate script:
```bash
python split_code15.py          # For CODE15 dataset
python split_ptbxl_samitrop.py  # For PTB-XL and SaMi-Trop datasets
```

4. Run the model pipeline: train, run, evaluate model 

```bash
python train_model.py -d data_train -m model -v

python run_model.py -d data_test -m model -o outputs -v

python evaluate_model.py -d data_test -o outputs -s scores.csv
```

---

### Project Structure

```
project_root/
│
├── data/                    # Raw and preprocessed WFDB data + metadata
│   ├── code15_output/
│   ├── ptbxl_output/
│   ├── samitrop_output/
│   ├── code15_meta.csv
│   └── ptbxl_meta.csv
├── data_train/              # Training data after split
├── data_test/               # Testing data after split
│
├── train_model.py           # Script to train model
├── run_model.py             # Script to run trained model on test set
├── evaluate_model.py        # Script to evaluate model predictions
├── helper_code.py           # Utility functions
│
├── team_code.py             # To switch between different models
├── model_xxx.py             # Individual model implementations (e.g., model_xgb.py, model_cnn.py)
│
├── model/                   # Saved trained models
├── outputs/                 # Model outputs (e.g., predictions on test set)
├── score/                   # Evaluation metrics
├── visualisation.ipynb      # Explore feature importance, Grad-CAM, etc.
│
├── split_code15.py          # Split CODE15 data
├── split_ptbxl_samitrop.py  # Split PTB-XL and SaMi-Trop data
│
├── requirements.txt         # Python dependencies
```

---

## Datasets

| Dataset | Records | Sampling | Duration | Label Type | Chagas Status |
|--------|---------|----------|----------|------------|----------------|
| [CODE-15%](https://zenodo.org/records/4916206) | 345,779 | 400 Hz | 7.3s / 10.2s | Weak (self-reported) | ~1.91% positive |
| [SaMi-Trop](https://zenodo.org/records/4905618) | 1,631 | 400 Hz | 7.3s / 10.2s | Strong (serological) | 100% positive |
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 21,799 | 500 Hz | 10s | Strong (geographic proxy) | 100% negative |

Each dataset is preprocessed into WFDB format following the instruction given by [PhysioNet](https://github.com/physionetchallenges/python-example-2025)

Information about patient-level identifiers (included in `meta.csv`) are used in train test split to prevent data leakage.

---
### WFDB File Format

Each ECG record includes:
- `record.hea`: Header file with metadata (sampling rate, channels, patient info)
- `record.dat`: Binary waveform data

Example from CODE-15% (Record 1300):

```
1300 12 400 2934
1300.dat 16 1000(0)/mV 16 0 -59 -6392 0 I
1300.dat 16 1000(0)/mV 16 0 -39 -8695 0 II
1300.dat 16 1000(0)/mV 16 0 20 -2273 0 III
1300.dat 16 1000(0)/mV 16 0 49 21456 0 AVR
1300.dat 16 1000(0)/mV 16 0 -39 4976 0 AVL
1300.dat 16 1000(0)/mV 16 0 -10 1401 0 AVF
1300.dat 16 1000(0)/mV 16 0 -78 14714 0 V1
1300.dat 16 1000(0)/mV 16 0 -235 4718 0 V2
1300.dat 16 1000(0)/mV 16 0 -244 -16992 0 V3
1300.dat 16 1000(0)/mV 16 0 -225 -7127 0 V4
1300.dat 16 1000(0)/mV 16 0 -156 30648 0 V5
1300.dat 16 1000(0)/mV 16 0 -156 12901 0 V6
# Age: 75
# Sex: Female
# Chagas label: True
# Source: CODE-15%
```

<details>
<summary>Header Explanation</summary>

- `1300 12 400 2934`: Record name, number of leads, sampling frequency (Hz), number of samples
- Signal line format:
  ```
  <file> <format> <gain>(baseline)/units <adc_res> <zero> <init_val> <checksum> <blk_size> <lead_name>
  ```
  Example:
  ```
  1300.dat 16 1000(0)/mV 16 0 -59 -6392 0 I
  ```

  - `1300.dat`: Signal data file
  - `16`: Format code (2-byte signed integers)
  - `1000(0)/mV`: Gain of 1000 units/mV and baseline of 0
  - `16`: ADC resolution (16 bits)
  - `0`: Zero reference value
  - `-59`: Initial value
  - `-6392`: Checksum
  - `0`: Block size
  - `I`: ECG Lead name

</details>