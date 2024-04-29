import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
import time
from tqdm import tqdm
import torchvision.transforms as transforms
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch.nn.functional as F


#### For backround removal (from hugging face) #####

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


###########

### For convolutional auto-encoder

def step_to_frame(step):
    return int(int(step)/10*59.95+5*59.95)
    
def read_video(id, view, type, needed_frames, backround_removal=False):
    """Reads video to numpy array using Open-CV"""
    if backround_removal:
        filepath = f"nfl-player-contact-detection/{type}/backround_removal/{id}_{view}.mp4"
    else:
        filepath = f"nfl-player-contact-detection/{type}/{id}_{view}.mp4"
    # Open the video file
    cap = cv2.VideoCapture(filepath)
    
    # Get video properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize an empty numpy array to store the frames
    video_array = np.empty((len(needed_frames), height, width), dtype=np.uint8)
    
    # Loop through each frame and store it in the numpy array
    i=0
    for request_frame in needed_frames:
        if (request_frame >= num_frames) or (request_frame < 0):
            video_array[i]=np.zeros((height, width))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, request_frame)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Frame not returned for play {id} frame {request_frame}. There are {num_frames} frames.")
        if frame.shape[0]>1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video_array[i]=frame
        i+=1
    # Release the video object
    cap.release()
    return video_array

def view_contact(video_array, helmet_mask):
    """Plots contact with helmet mask."""
    plt.imshow(video_array)
    plt.imshow(helmet_mask, alpha=0.5, cmap='Reds')  
    plt.show()
    plt.close()

def create_boxes_dict(id, view, array_size, helmet_df, player_id, frames):
    """Creates array of helmet mask boxes from helmet tracking data."""
    # Query the DataFrame once for all frames
    player_id=int(player_id)
    frame_data = helmet_df.query("view==@view & game_play==@id & nfl_player_id==@player_id")
    frames_dict = dict()
    for frame_idx in frames:
        if frame_idx in frame_data.frame.values:
            left, top, width, height = frame_data.loc[frame_data.frame==frame_idx, ['left', 'top', 'width', 'height']].reset_index(drop=1).iloc[0,]
            # Ensure box coordinates are within the array bounds
            left = max(0, left)
            top = max(0, top)
            right = min(array_size[1], left + width)
            bottom = min(array_size[0], top + height)

            # Mark the region of the box in the numpy array
            boxes_array = np.zeros(array_size, dtype=np.uint8)
            boxes_array[top:bottom, left:right] = 255
        else:
            boxes_array = np.zeros(array_size, dtype=np.uint8)
        frames_dict.update({frame_idx : boxes_array})
    return frames_dict

class ContactDataset:
    """Dataset for training and testing zoomed contact examples."""

    # TO DO: Cross validate image size, add multiple frames per play, cross validate how many plays back forward, cross val skips

    def __init__(self, record_df_path, ground = False, feature_size=256, num_back_forward_steps=2, skips=1, distance_cutoff=5,
                N=10000, pos_balance = 0.5, backround_removal=False):
            
        self.ground=ground
        if ground:
            self.record_df = pd.read_csv(record_df_path).query("nfl_player_id_2 == 'G'")
        else:
            self.record_df = pd.read_csv(record_df_path).query("nfl_player_id_2 != 'G'")
        if "train" in record_df_path:
            self.type = "train"
        else:
            self.type = "test"

        self.tracking_df = pd.read_csv(os.getcwd() + f"/nfl-player-contact-detection/{self.type}_player_tracking.csv")
        self.helmets_df = pd.read_csv(os.getcwd() + f"/nfl-player-contact-detection/{self.type}_baseline_helmets.csv")
        self.feature_size = feature_size
        self.skips=skips
        self.num_back_forward_steps=num_back_forward_steps
        self.distance_cutoff = distance_cutoff
        self.cache=dict()
        self.backround_removal=backround_removal

        if not self.ground:
            # Filter to only plays with cutoff distance (others will be assigned 0 contact prob)

            loc_df = self.tracking_df[["game_play", "nfl_player_id", "step", "x_position", "y_position"]]
            merge_p1 = (
                self.record_df.astype(str).merge(
                    loc_df.rename(
                        {"nfl_player_id": "nfl_player_id_1", "x_position": "x_position_1", "y_position": "y_position_1"},
                        axis=1,
                    ).astype(str),
                    on=["game_play", "nfl_player_id_1", "step"],
                    how="left",
                )
                .merge(
                    loc_df.rename(
                        {"nfl_player_id": "nfl_player_id_2", "x_position": "x_position_2", "y_position": "y_position_2"},
                        axis=1,
                    ).astype(str),
                    on=["game_play", "nfl_player_id_2", "step"],
                    how="left",
                )
            )
            merge_p1['distance']=(merge_p1.x_position_1.astype(float)-merge_p1.x_position_2.astype(float))**2+(merge_p1.y_position_1.astype(float)-merge_p1.y_position_2.astype(float))**2
            merge_p1 = merge_p1.loc[merge_p1.distance < self.distance_cutoff]
            self.record_df = self.record_df.loc[self.record_df.contact_id.isin(merge_p1.contact_id)]
            self.record_df['frame'] = list(map(step_to_frame, self.record_df.step))
            self.record_df['nfl_player_id_1'] = self.record_df['nfl_player_id_1'].astype(int)
            self.record_df['nfl_player_id_2'] = self.record_df['nfl_player_id_2'].astype(int)
        
        # Filter to balanced sample
        if not self.type == "test":
            self.pos_class = self.record_df.query("contact == 1").sample(n=int(N*pos_balance), replace=False, random_state=1).reset_index(drop=1)
            self.neg_class = self.record_df.query("contact == 0").sample(n=int(N*(1-pos_balance)), replace=False, random_state=1).reset_index(drop=1)
            self.record_df  = pd.concat([self.pos_class, self.neg_class], axis = 0).reset_index(drop = 1)
            print(f"Data Sample Contains {self.record_df.shape[0]} observations.")
            
    def get_features(self, contact_info_df, box_cache=None):
        """Gets features from single row of records df."""
        label = int(contact_info_df['contact'])
        game_play = contact_info_df['game_play']
        player_1_id = int(contact_info_df['nfl_player_id_1'])
        player_2_id = "G" if self.ground else int(contact_info_df['nfl_player_id_2'])
        step = contact_info_df['step']
        frame_id = step_to_frame(step)
        steps = np.arange(step-self.num_back_forward_steps*self.skips, 
                        step+self.num_back_forward_steps*self.skips+1, self.skips)
        frame_ids = np.sort([step_to_frame(x) for x in steps])

        half_feature_size = self.feature_size // 2

        # Get distance info (if not ground play)
        if not self.ground:
            p1_row_track = self.tracking_df.loc[(self.tracking_df.game_play==game_play) & 
                                                (self.tracking_df.step.isin(steps)) & 
                                                (self.tracking_df.nfl_player_id==player_1_id)]
            p2_row_track = self.tracking_df.loc[(self.tracking_df.game_play==game_play) & 
                                                (self.tracking_df.step.isin(steps)) & 
                                                (self.tracking_df.nfl_player_id==player_2_id)]
            distance = np.sqrt((p1_row_track['x_position'].values - p2_row_track['x_position'].values)**2 + 
                            (p1_row_track['y_position'].values - p2_row_track['y_position'].values)**2)
            if len(distance) != len(steps):
                found_index=p1_row_track.step.values-min(steps)
                fixed = np.zeros(len(steps))
                fixed[found_index] = distance
                distance=fixed.copy()
            distance_as_mat = np.full((1, len(distance), self.feature_size, self.feature_size), distance[:, None, None])

        # Get video arrays and helmet masks
        video_arrays = []
        mask_arrays = []
        centerpoints = {}
        i=0
        for view in ["Sideline", "Endzone"]:
            # Video array
            raw_frames = read_video(id=game_play, view=view, type=self.type, 
                                        needed_frames=frame_ids, backround_removal=self.backround_removal)

            # Pad
            dim_1, dim_2=raw_frames.shape[1], raw_frames.shape[2]
            raw_frames=np.pad(raw_frames, pad_width=[(0, 0)] + [(half_feature_size, half_feature_size)] * 2, mode='constant', constant_values=0)

            # Helmet masks
            helmet_mask_player_1_dict = create_boxes_dict(id=game_play, view=view, array_size = (dim_1, dim_2),
                                                        helmet_df=self.helmets_df, player_id = player_1_id, frames = frame_ids)
            helmet_mask_player_2_dict = create_boxes_dict(id=game_play, view=view, array_size = (dim_1, dim_2),
                                                        helmet_df=self.helmets_df, player_id = player_2_id, frames = frame_ids)
            helmet_masks_player_1 = np.stack([helmet_mask_player_1_dict[frame_id] for frame_id in frame_ids])
            helmet_masks_player_2 = np.stack([helmet_mask_player_2_dict[frame_id] for frame_id in frame_ids])  
            helmet_mask_frames = helmet_masks_player_1 + helmet_masks_player_2
            helmet_mask_frames = np.pad(helmet_mask_frames, pad_width=[(0, 0)] + [(half_feature_size, half_feature_size)] * 2, mode='constant', constant_values=0)
            
            # Centerpoints & Zoom
            helmet_mask_df = self.helmets_df.query("view==@view & game_play==@game_play & frame==@frame_id")
            df_this_frame = helmet_mask_df.loc[helmet_mask_df['nfl_player_id'].isin([player_1_id, player_2_id])]
            if df_this_frame.empty:
                centerpoint = (raw_frames.shape[1] // 2, raw_frames.shape[2] // 2)
            else:
                x = np.mean(df_this_frame['left'].values + (df_this_frame['width'].values / 2))
                y = np.mean(df_this_frame['top'].values - (df_this_frame['height'].values / 2))
                if x < 0:
                    x = 0
                elif x > raw_frames.shape[2]:
                    x = raw_frames.shape[2]
                if y < 0:
                    y = 0
                elif y > raw_frames.shape[1]:
                    y = raw_frames.shape[1]
                centerpoint = (int(x) + half_feature_size, int(y) + half_feature_size)
            mask_arrays.append(helmet_mask_frames[:, (centerpoint[1]-half_feature_size):(centerpoint[1]+half_feature_size), (centerpoint[0]-half_feature_size):(centerpoint[0]+half_feature_size)])
            video_arrays.append(raw_frames[:, (centerpoint[1]-half_feature_size):(centerpoint[1]+half_feature_size), (centerpoint[0]-half_feature_size):(centerpoint[0]+half_feature_size)])
            i+=1
        video_arrays = np.stack(video_arrays, axis = 0) # 2 (Views), num_frames, feature_size, fs
        mask_arrays = np.stack(mask_arrays, axis = 0)  # 2 (Views), num_frames, feature_size, fs

        # Organize
        feature = [video_arrays, mask_arrays]
        if not self.ground:
            feature.append(distance_as_mat)
        feature_array = np.concatenate(feature, axis = 0) # Channel, Time, H, W
        ret = (tuple(torch.Tensor(feature_array[:, timestep, :, :]) for timestep in range(feature_array.shape[1])), torch.Tensor([label]))
        return ret

    def __getitem__(self, idx):
        # Organize record info
        if idx in self.cache.keys():
            return self.cache[idx]
        contact_info_df = self.record_df.iloc[idx,:]
        self.cache.update({idx : self.get_features(contact_info_df)}) 
        return self.cache[idx]

    def __len__(self):
        return len(self.record_df)
    
#### NEED TO IMPLIMENT TRANSOFRMS FOR 5 CHANNEL IMAGE 
tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop the image and resize to 224x224
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
            transforms.RandomRotation(degrees=30)
])