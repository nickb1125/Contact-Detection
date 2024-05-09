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
import random
import polars as pl


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
    
def read_video(id, view, type, needed_frames):
    """Reads video to numpy array using Open-CV"""
    filepath_bases = [f"nfl-player-contact-detection/{type}/frames/{id}_{view}_{frame}.jpg" for frame in needed_frames]
    frames = []
    for path in filepath_bases:
        image = cv2.imread(path)  # Read image in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frames.append(image)
    frames_tensor = np.stack(frames, axis=0)
    return frames_tensor

def view_contact(video_array, helmet_mask):
    """Plots contact with helmet mask."""
    plt.imshow(video_array)
    plt.imshow(helmet_mask, alpha=0.5, cmap='Reds')  
    plt.show()
    plt.close()

def create_boxes_dict(id, view, array_size, helmet_df, player_id, frames):
    """Creates array of helmet mask boxes from helmet tracking data."""
    
    # Query the DataFrame once for all frames
    player_id = str(player_id)
    frame_data = helmet_df.filter(
        (pl.col("view") == view)
        & (pl.col("game_play") == id)
        & (pl.col("nfl_player_id") == player_id)
    )
    frames_dict = {}
    for frame_idx in frames:
        frame_idx = int(frame_idx)
        if frame_idx in frame_data["frame"].to_numpy():
            # Get box coordinates for the current frame
            frame_row = frame_data.filter(pl.col("frame") == frame_idx)
            left = frame_row["left"][0]
            top = frame_row["top"][0]
            width = frame_row["width"][0]
            height = frame_row["height"][0]

            # Ensure box coordinates are within the array bounds
            left = max(0, left)
            top = max(0, top)
            right = min(array_size[1], left + width)
            bottom = min(array_size[0], top + height)

            # Create a new numpy array for the boxes
            boxes_array = np.zeros(array_size, dtype=np.uint8)
            boxes_array[top:bottom, left:right] = 255
        else:
            # No box for this frame, create an empty array
            boxes_array = np.zeros(array_size, dtype=np.uint8)

        frames_dict[frame_idx] = boxes_array

    return frames_dict

class ContactDataset:
    """Dataset for training and testing zoomed contact examples."""

    def __init__(self, record_df_path, feature_size=256, num_back_forward_steps=2, skips=1, distance_cutoff=5,
                 N=10000, pos_balance=0.5, background_removal=False):

        self.record_df = pl.read_csv(record_df_path, dtypes={"nfl_player_id_1" : str, "nfl_player_id_2" : str}).\
            filter(pl.col("nfl_player_id_2") != "G")
        if "train" in record_df_path:
            self.type = "train"
        else:
            self.type = "test"
        
        tracking_df_path = os.path.join(os.getcwd(), f"nfl-player-contact-detection/{self.type}_player_tracking.csv")
        helmets_df_path = os.path.join(os.getcwd(), f"nfl-player-contact-detection/{self.type}_baseline_helmets.csv")
        
        self.tracking_df = pl.read_csv(tracking_df_path, dtypes={"nfl_player_id" : str})
        self.helmets_df = pl.read_csv(helmets_df_path, dtypes={"nfl_player_id" : str})
        
        self.feature_size = feature_size
        self.skips = skips
        self.num_back_forward_steps = num_back_forward_steps
        self.distance_cutoff = distance_cutoff
        self.background_removal = background_removal
        self.half_feature_size = self.feature_size // 2
        
        # Filter to only plays with cutoff distance (others will be assigned 0 contact prob)
        loc_df = self.tracking_df.select([
            "game_play", "nfl_player_id", "step", "x_position", "y_position"
        ])
        
        merge_p1 = (
            self.record_df.join(
                loc_df.select(
                    pl.col("game_play"),
                    pl.col("step"),
                    pl.col("nfl_player_id").alias("nfl_player_id_1"),
                    pl.col("x_position").alias("x_position_1"),
                    pl.col("y_position").alias("y_position_1")
                ),
                on=["game_play", "nfl_player_id_1", "step"],
                how="left"
            )
            .join(
                loc_df.select(
                    pl.col("game_play"),
                    pl.col("step"),
                    pl.col("nfl_player_id").alias("nfl_player_id_2"),
                    pl.col("x_position").alias("x_position_2"),
                    pl.col("y_position").alias("y_position_2")
                ),
                on=["game_play", "nfl_player_id_2", "step"],
                how="left"
            )
        )
        
        merge_p1 = merge_p1.with_columns(
            ((pl.col("x_position_1").cast(float) - pl.col("x_position_2").cast(float))**2 +
            (pl.col("y_position_1").cast(float) - pl.col("y_position_2").cast(float))**2).alias("distance")
        )
        
        merge_p1 = merge_p1.filter(
            (pl.col("distance") < self.distance_cutoff) | pl.col("distance").is_null()
        )
        
        self.record_df = self.record_df.filter(
            pl.col("contact_id").is_in(merge_p1['contact_id'].to_numpy()))
        
        self.record_df = self.record_df.with_columns(
            pl.col("step").apply(step_to_frame).cast(int).alias('frame'),
        )
        
        # Filter to balanced sample with proper play number
        if self.type != "test":
            # Randomly select N unique plays
            unique_groups = self.record_df["game_play"].unique().to_numpy()
            if N > len(unique_groups):
                print("N plays is greater than number of plays in data, using entire set....")
                N = len(unique_groups)
            selected_groups = np.random.choice(unique_groups, size=N, replace=False)
            self.record_df = self.record_df.filter(pl.col("game_play").is_in(selected_groups))
            
            # Balance positive and negative samples
            pos_df = self.record_df.filter(pl.col("contact") == 1)
            neg_df = self.record_df.filter(pl.col("contact") == 0)
            
            num_pos = pos_df.shape[0]
            num_neg = neg_df.shape[0]
            
            
            if num_pos > num_neg:
                self.pos_class = pos_df.sample(n=num_neg, with_replacement=False, seed=1)
                self.neg_class = neg_df
            else:
                self.pos_class = pos_df
                self.neg_class = neg_df.sample(n=num_pos, with_replacement=False, seed=1)
            
            # Achieve proper balance
            self.pos_class = self.pos_class.sample(fraction=pos_balance, with_replacement=False, seed=1)
            
            # Combine
            self.record_df = pl.concat([self.pos_class, self.neg_class], how = "vertical")
            self.game_plays = self.record_df["game_play"].unique().to_numpy()
            print(f"Data Sample Contains {self.record_df.shape[0]} observations.")
        
    def get_features(self, contact_info_df, video_cache=None, box_cache=None):
        """Gets features from single row of records df."""
        label = contact_info_df['contact'][0]
        if not np.isnan(label):
            label = int(label)
        game_play = contact_info_df['game_play'][0]
        contact_id = contact_info_df['contact_id'][0]
        player_1_id = str(contact_info_df['nfl_player_id_1'][0])
        player_2_id = contact_info_df['nfl_player_id_2'][0]
        step = contact_info_df['step'][0]
        frame_id = step_to_frame(step)
        steps = list(range(step - self.num_back_forward_steps * self.skips,
                           step + self.num_back_forward_steps * self.skips + 1, self.skips))
        frame_ids = [step_to_frame(x) for x in steps]

        # Get distance info 
        if player_2_id != "G":
            p1_row_track = self.tracking_df.filter(
                (pl.col("game_play") == game_play)
                & (pl.col("step").is_in(steps))
                & (pl.col("nfl_player_id") == player_1_id)
            )
            p2_row_track = self.tracking_df.filter(
                (pl.col("game_play") == game_play)
                & (pl.col("step").is_in(steps))
                & (pl.col("nfl_player_id") == player_2_id)
            )

            missing_steps = set(steps) - set(p1_row_track.select("step").to_numpy().flatten())
            distance = np.sqrt(
                (p1_row_track.select("x_position").to_numpy().flatten() - p2_row_track.select("x_position").to_numpy().flatten()) ** 2 +
                (p1_row_track.select("y_position").to_numpy().flatten() - p2_row_track.select("y_position").to_numpy().flatten()) ** 2
            )

            if missing_steps:
                distance = np.array([0 if step in missing_steps else distance[steps.index(step)] for step in steps])

            distance_as_mat = np.full((1, len(distance), self.feature_size, self.feature_size), distance[:, None, None])
        else:
            distance_as_mat = np.zeros((1, len(steps), self.feature_size, self.feature_size))

        # Get video arrays and helmet masks
        video_arrays = []
        mask_arrays = []
        for view in ["Sideline", "Endzone"]:
            # Video array
            if video_cache is not None:
                cache_view = video_cache[view]
                option_replace = np.zeros(cache_view[0].shape)
                raw_frames = np.stack([cache_view[frame, :, :] if frame < cache_view.shape[0] else option_replace for frame in frame_ids], axis=0)
            else:
                raw_frames = read_video(id=game_play, view=view, type=self.type, needed_frames=frame_ids)
                
            # Pad
            raw_frames = np.pad(raw_frames, pad_width= [(0, 0)] + [(self.half_feature_size, self.half_feature_size)] * 2, 
                                mode='constant', constant_values=0)

            # Helmet masks
            if box_cache is not None:
                helmet_mask_player_1_dict = box_cache[view][player_1_id]
                helmet_mask_player_2_dict = box_cache[view][player_2_id]
            else:
                helmet_mask_player_1_dict = create_boxes_dict(id=game_play, view=view, array_size=(raw_frames.shape[1], raw_frames.shape[2]),
                                                              helmet_df=self.helmets_df, player_id=player_1_id, frames=frame_ids)
                helmet_mask_player_2_dict = create_boxes_dict(id=game_play, view=view, array_size=(raw_frames.shape[1], raw_frames.shape[2]),
                                                              helmet_df=self.helmets_df, player_id=player_2_id, frames=frame_ids)

            helmet_masks_player_1 = np.stack([helmet_mask_player_1_dict[frame_id] for frame_id in frame_ids])
            helmet_masks_player_2 = np.stack([helmet_mask_player_2_dict[frame_id] for frame_id in frame_ids])  
            helmet_mask_frames = helmet_masks_player_1 + helmet_masks_player_2
            helmet_mask_frames = np.pad(helmet_mask_frames, pad_width=[(0, 0)] + [(self.half_feature_size, self.half_feature_size)] * 2, mode='constant', constant_values=0)

            # Centerpoints & Zoom
            helmet_mask_df = self.helmets_df.filter(
                (pl.col("view") == view)
                & (pl.col("game_play") == game_play)
                & (pl.col("frame") == frame_id)
            )
            df_this_frame = helmet_mask_df.filter(pl.col("nfl_player_id").is_in([player_1_id, player_2_id]))
            if df_this_frame.shape[0] == 0:
                centerpoint = (raw_frames.shape[1] // 2, raw_frames.shape[2] // 2)
            else:
                x = np.mean(df_this_frame.select("left").to_numpy().flatten() + (df_this_frame.select("width").to_numpy().flatten() / 2))
                y = np.mean(df_this_frame.select("top").to_numpy().flatten() - (df_this_frame.select("height").to_numpy().flatten() / 2))
                x = max(0, min(x, raw_frames.shape[2]))
                y = max(0, min(y, raw_frames.shape[1]))
                centerpoint = (int(x) + self.half_feature_size, int(y) + self.half_feature_size)

            mask_arrays.append(helmet_mask_frames[:, (centerpoint[1] - self.half_feature_size):(centerpoint[1] + self.half_feature_size), (centerpoint[0] - self.half_feature_size):(centerpoint[0] + self.half_feature_size)])
            video_arrays.append(raw_frames[:, (centerpoint[1] - self.half_feature_size):(centerpoint[1] + self.half_feature_size), (centerpoint[0] - self.half_feature_size):(centerpoint[0] + self.half_feature_size)])

        video_arrays = np.stack(video_arrays, axis=0)  # 2 (Views), num_frames, feature_size, fs
        mask_arrays = np.stack(mask_arrays, axis=0)    # 2 (Views), num_frames, feature_size, fs

        # Organize
        feature = [video_arrays, mask_arrays, distance_as_mat]
        feature_array = np.concatenate(feature, axis = 0) # Channel, Time, H, W
        ret = (tuple(torch.Tensor(feature_array[:, timestep, :, :]) for timestep in range(feature_array.shape[1])), torch.Tensor([label]))
        return ret

    def __getitem__(self, idx):
        # Organize record info
        contact_info_df = self.record_df[idx,:]
        return self.get_features(contact_info_df)

    def __len__(self):
        return len(self.record_df)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#### NEED TO IMPLIMENT TRANSOFRMS FOR 5 CHANNEL IMAGE 
tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop the image and resize to 224x224
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
            transforms.RandomRotation(degrees=30)
])



