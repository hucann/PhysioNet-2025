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
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):

    # Load records from the data folder
    train_records = collect_records(os.path.join(data_folder, "train"), ["ptbxl_output", "samitrop_output"])
    val_records   = collect_records(os.path.join(data_folder, "val"),   ["ptbxl_output", "samitrop_output"])

    train_dataset = ECGDataset(train_records, os.path.join(data_folder, "train"))
    val_dataset = ECGDataset(val_records, os.path.join(data_folder, "val"))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Select device: MPS for Mac GPU, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    model = ECG1DCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for signals, labels in loop:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = (outputs.squeeze() > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)

                outputs = model(signals)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                predictions = (outputs.squeeze() > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / val_total)

        if verbose:
            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_losses[-1]:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_accuracies[-1]:.4f}")

    os.makedirs(model_folder, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(model_folder, 'cnn_model.pth'))

    # Plot loss against epochs
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig(os.path.join(model_folder, "loss_plot.png"))
    plt.close()

    # Plot accuracy against epochs
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")
    plt.savefig(os.path.join(model_folder, "accuracy_plot.png"))
    plt.close()

    if verbose:
        print('Training complete, model and plots saved.')

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
    def __init__(self, records_with_source, data_folder):
        self.records_with_source = records_with_source  # list of (record_name, subfolder)
        self.data_folder = data_folder

    def __len__(self):
        return len(self.records_with_source)

    def __getitem__(self, idx):
        record_name, subfolder = self.records_with_source[idx]
        record_path = os.path.join(self.data_folder, subfolder, record_name)

        signal, _ = load_signals(record_path)
        label = load_label(record_path)

        # Normalize and pad to fixed length
        signal = np.nan_to_num(signal).T
        signal = signal[:, :5000]
        if signal.shape[1] < 5000:
            signal = np.pad(signal, ((0, 0), (0, 5000 - signal.shape[1])), mode='constant')

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
def collect_records(base_dir, subfolders):
    all_records = []
    for subfolder in subfolders:
        full_path = os.path.join(base_dir, subfolder)
        records = find_records(full_path)
        all_records += [(record, subfolder) for record in records]
    return all_records