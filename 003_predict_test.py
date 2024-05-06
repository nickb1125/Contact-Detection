import pandas as pd
import os
from models import ContactNet
from objects import ContactDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import math


###### Connect to device 

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print("CUDA is available! Using GPU.")
    torch.cuda.init()
else:
    device = torch.device("cpu")  # Use CPU

# Load Settings
print("----Loading Settings-----")

settings_df = pd.read_csv(os.getcwd() + '/models/settings.csv')
num_back_forward_steps = settings_df.num_back_forward_steps.values[0]
skips= settings_df.skips.values[0]
distance_cutoff= settings_df.distance_cutoff.values[0]
image_size = settings_df.image_size.values[0]
input_size =  settings_df.input_size.values[0]
hidden_size =  settings_df.hidden_size.values[0]
num_layers =  settings_df.num_layers.values[0]
dropout =  settings_df.dropout.values[0]

test_info = pd.read_csv(os.getcwd() + "/nfl-player-contact-detection/test_labels.csv")

# Load trained model

print("----Loading Trained Model-----")
model = ContactNet(image_size, input_size, hidden_size, num_layers, dropout)
state_dict = torch.load(os.getcwd() + '/models/contact_model.pth')
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Create dataset

print("----Creating Dataset and Cacheing Features-----")
dataset = ContactDataset(os.getcwd() + "/nfl-player-contact-detection/test_labels.csv",
                      feature_size=image_size, num_back_forward_steps=num_back_forward_steps, 
                      skips=skips, distance_cutoff=distance_cutoff, N=np.NaN, pos_balance=np.NaN)
dataset._cache_all_features()
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

# Predict
print("----Predicting-----")
all_predictions = []
all_labels = []
with torch.no_grad():  # Disable gradient tracking during inference
    for batch_idx, (features, labels) in tqdm(enumerate(dataloader), total=math.ceil(len(dataset)/256)):
        x1, x2, x3 = features
        x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)

        # Forward pass (inference) to obtain predictions
        outputs = model(x1, x2, x3)
        
        # Collect predictions and labels
        all_predictions.extend(outputs.cpu().numpy())

print("----Saving-----")
all_predictions_within_distance = np.array(all_predictions)
pred_df = dataset.record_df[["contact_id"]]
contact_values = [1 if x > 0.5 else 0 for x in all_predictions_within_distance]
pred_df.loc[:, "contact"] = contact_values
submission_df = test_info[["contact_id"]].merge(pred_df, how = "left", on = "contact_id")
contact_values_complete = [0 if np.isnan(x) else x for x in submission_df.contact]
submission_df.loc[:, "contact"] = contact_values_complete
submission_df.to_csv(os.getcwd() + '/submission.csv')

print(f"Predicted {sum(submission_df.contact)} contacts out of {len(submission_df)} observations...")