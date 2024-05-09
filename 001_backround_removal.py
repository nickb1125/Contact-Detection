"""For removal of all frame backrounds in train and test sets"""

from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
from objects import preprocess_image, postprocess_image, step_to_frame
import cv2
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import chain
import shutil

####### SETTING #######

backround_remove=False

####################

# Get frames requested in train and test

train_info = pd.read_csv(os.getcwd() + "/nfl-player-contact-detection/train_labels.csv")
needed_train_df = train_info[['game_play', 'step']].drop_duplicates().reset_index(drop=1)
needed_train_df['frame'] = list(map(step_to_frame, needed_train_df.step))

test_info=pd.read_csv(os.getcwd() + "/nfl-player-contact-detection/sample_submission.csv")
test_info["game_play"] = list(map(lambda text: text.split("_")[0] + "_" + text.split("_")[1], test_info["contact_id"]))
test_info["step"] = list(map(lambda text: text.split("_")[2], test_info["contact_id"]))
test_info["nfl_player_id_1"] = list(map(lambda text: text.split("_")[3], test_info["contact_id"]))
test_info["nfl_player_id_2"] = list(map(lambda text: text.split("_")[4], test_info["contact_id"]))
test_info["contact"]=np.NaN
test_info.to_csv(os.getcwd() + "/nfl-player-contact-detection/test_labels.csv")
needed_test_df = test_info[['game_play', 'step']].drop_duplicates().reset_index(drop=1)
needed_test_df['frame'] = list(map(step_to_frame, needed_test_df.step))

if os.path.exists(os.getcwd() + "/nfl-player-contact-detection/train/frames"):
    shutil.rmtree(os.getcwd() + "/nfl-player-contact-detection/train/frames")
if os.path.exists(os.getcwd() + "/nfl-player-contact-detection/test/frames"):
    shutil.rmtree(os.getcwd() + "/nfl-player-contact-detection/test/frames")

# Get train video filepaths
train_file_paths = []
for root, dirs, files in os.walk(os.getcwd() + "/nfl-player-contact-detection/train"):
    for file in files:
        train_file_paths.append(os.path.join(root, file))

# Get test video filepaths
test_file_paths = []
for root, dirs, files in os.walk(os.getcwd() + "/nfl-player-contact-detection/test"):
    for file in files:
        test_file_paths.append(os.path.join(root, file))

# Combine to get all video file paths

train_file_paths.extend(test_file_paths)
all_file_paths=train_file_paths.copy()
all_file_paths = [item for item in all_file_paths if ('Sideline' in item) or ("Endzone" in item)] 

print(f"----Saving frames from {len(needed_train_df.game_play.unique())} plays and {len(all_file_paths)} views.-----")

if backround_remove:
    # Pull pretrained backround removal model from hugging face
    print("----Loading Backround Removal Pretrained Model------")
    backround_removal_model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print("CUDA is available! Using GPU.")
        torch.cuda.init()
    else:
        device = torch.device("cpu")  # Use CPU
        print("CUDA is not available. Using CPU.")
    backround_removal_model.to(device)

# Make save directories if needed

os.makedirs(os.getcwd() + "/nfl-player-contact-detection/train/frames", exist_ok=True)
os.makedirs(os.getcwd() + "/nfl-player-contact-detection/test/frames", exist_ok=True)

# Get video save options

print("------Saving Backround Removed Copies---")

for filepath in tqdm(all_file_paths):

    # Open the video file
    cap = cv2.VideoCapture(filepath)

    # Get video properties
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get information for save
    base = filepath.split("/")[-1].split(".")[0]
    try:
        game_play = base.split("_")[0]+ "_"+ base.split("_")[1]
    except:
        print(base)
        raise ValueError
    if "train" in filepath:
        output_file_path = os.getcwd() + f"/nfl-player-contact-detection/train/frames/{base}"
        needed_frames = set(needed_train_df.loc[needed_train_df.game_play==game_play].frame.values)
    else:
        output_file_path = os.getcwd() + f"/nfl-player-contact-detection/test/frames/{base}"
        needed_frames = set(needed_test_df.loc[needed_test_df.game_play==game_play].frame.values)
    
    # Add 20 closest frames to needed frame (for temporal needs)
    needed_frames = set(list(chain.from_iterable([list(range(x-10, x+10)) for x in needed_frames])))

    # Loop through each frame and store it in the numpy array
    for request_frame in needed_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, request_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        if backround_remove:
            orig_im_size = frame.shape[0:2]
            image = preprocess_image(frame, orig_im_size).to(device)
            result=backround_removal_model(image)
            frame = postprocess_image(result[0][0], orig_im_size)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Save frame
        cv2.imwrite(output_file_path + "_" + str(request_frame) + ".jpg", frame)
        
    # Release the video object
    cap.release()
    
