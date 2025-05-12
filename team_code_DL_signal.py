#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import sys

from helper_code import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print(f'Found {num_records} records.')
        print('Preparing dataset...')

    dataset = ECGDataset(records, data_folder)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    model = ECG1DCNN(in_channels=12)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for signals, labels in loop:
            outputs = model(signals)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(model_folder, 'cnn_model.pth'))

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model = ECG1DCNN()
    model_path = os.path.join(model_folder, 'cnn_model.pth')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    signal, _ = load_signals(record)
    signal = np.nan_to_num(signal).T  # (channels, time)
    signal = signal[:, :5000]
    if signal.shape[1] < 5000:
        signal = np.pad(signal, ((0, 0), (0, 5000 - signal.shape[1])), mode='constant')
    
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # Add batch dim
    output = model(signal).item()
    binary_output = int(output > 0.5)
    probability_output = output

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################  

# Define 1D CNN model
class ECG1DCNN(nn.Module):
    def __init__(self, in_channels=12, seq_len=5000):
        super(ECG1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x
    
# Create a custom PyTorch dataset
class ECGDataset(Dataset):
    def __init__(self, records, data_folder):
        self.records = records
        self.data_folder = data_folder
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record_path = os.path.join(self.data_folder, self.records[idx])
        signal, _ = load_signals(record_path)
        label = load_label(record_path)

        # Normalize and pad to fixed length
        signal = np.nan_to_num(signal)
        signal = signal.T  # Convert to shape (channels, time)
        signal = signal[:, :5000]  # Truncate or pad
        if signal.shape[1] < 5000:
            pad = 5000 - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad)), mode='constant')
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)