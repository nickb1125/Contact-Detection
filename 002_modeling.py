import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from objects import step_to_frame, create_boxes_dict, ContactDataset, seed_everything
from models import Encoder, ContactNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from sklearn.model_selection import train_test_split
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize

seed_everything(2)# Seed everything

# srun -p gpu-common  --gres=gpu --mem=4G --pty bash -i

############ SETTINGS #################

feature_size = 256 # Size of square input channels
num_back_forward_steps = 1 # Number of forward and backward timesteps
skips = 1 # How many steps between time steps 
distance_cutoff = 5 # Yard cutoff to set contact prob to zero
N = 100 # Number per contact label for balanced training sample
positive_allocation_rate = 0.3


image_size = feature_size
input_size = 100  # Output size of the Encoder's fully connected layer for lstm features
hidden_size = 64 # Size of lstm hidden layers
num_layers = 2 # Number of lstm layers
dropout = 0.1
learning_rate = 0.001
num_epochs = 10
val_size=0.2

#######################################

#### Train test splits #####

print("----Initiating Train/Validation Splits------")

# Manufacture and split train labels from val labels

train_val_labels = pd.read_csv(os.getcwd() + "/nfl-player-contact-detection/train_labels.csv")
train_df, val_df = train_test_split(train_val_labels, test_size=val_size, random_state=42)
train_df.to_csv(os.getcwd() + "/nfl-player-contact-detection/train_only_labels.csv")
val_df.to_csv(os.getcwd() + "/nfl-player-contact-detection/train_val_only_labels.csv")

###### Load hugging face backround removal model 

print("---Loading Backround Removal Pretrained Model----")
backround_model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)


###### Connect to device 

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print("CUDA is available! Using GPU.")
    torch.cuda.init()
else:
    device = torch.device("cpu")  # Use CPU
    print("CUDA is not available. Using CPU.")

backround_model.to(device)

print("---Loading Train Dataloader----")
dataset = ContactDataset(os.getcwd() + "/nfl-player-contact-detection/train_only_labels.csv",
                      ground=False, feature_size=feature_size, num_back_forward_steps=num_back_forward_steps, 
                      skips=skips, distance_cutoff=distance_cutoff, N=N, pos_balance=positive_allocation_rate)
print(f"-----Caching train features and labels-----")
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


print("---Loading Val Dataloader----")
val_dataset = ContactDataset(os.getcwd() + "/nfl-player-contact-detection/train_val_only_labels.csv",
                      ground=False, feature_size=feature_size, num_back_forward_steps=num_back_forward_steps, 
                      skips=skips, distance_cutoff=distance_cutoff, N=N*val_size, pos_balance=positive_allocation_rate)
print(f"-----Caching test features and labels-----")
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=True)


print("---Initializing Model----")
combined_model = ContactNet(image_size, input_size, hidden_size, num_layers, dropout)
combined_model.to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)


print("---Training----")
for epoch in range(num_epochs):
    # Training loop
    combined_model.train()
    for batch_idx, (features, labels) in enumerate(dataloader):
        x1, x2, x3 = features
        # Forward pass
        outputs = combined_model(x1, x2, x3)

        loss = criterion(outputs.squeeze(), labels.squeeze().float())  # Compute loss
        print(loss)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        # Print loss
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Validation loop
    combined_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for batch_idx, (test_features, test_labels) in enumerate(val_dataloader):
            x1_test, x2_test, x3_test = test_features
            # Forward pass
            outputs_test = combined_model(x1_test, x2_test, x3_test)

            loss_test = criterion(outputs_test.squeeze(), test_labels.squeeze().float())  # Compute loss

            total_loss += loss_test.item() * len(test_labels)
            total_samples += len(test_labels)

        average_loss = total_loss / total_samples
        print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')

os.makedirs(os.getcwd() + "/models", exist_ok=True)
torch.save(combined_model.state_dict(), os.getcwd() + '/models/contact_model.pth')
