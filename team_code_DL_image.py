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

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import timm
from tqdm import tqdm
import torch.nn as nn

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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ECGImageDataset(records, data_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}. Training loss: {epoch_loss / len(loader):.4f}")

    os.makedirs(model_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_folder, 'Transfer.pt'))

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(os.path.join(model_folder, "Transfer.pt"), map_location="cpu"))
    model.eval()
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    image_path = record + '.png'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)

    return pred, prob

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################  

class ECGImageDataset(Dataset):
    def __init__(self, records, data_folder, transform=None):
        self.records = records
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # record_path = self.records[idx]
        record_path = os.path.join(self.data_folder, self.records[idx])
        image_path = record_path + '.png'
        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(load_label(record_path), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label

