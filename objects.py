import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
import time

cache=dict()
helmet_cache=dict()

class VideoDataset(torch.utils.data.Dataset): # Note: This is not for contact classification, but for future work on limb tracking
    """Dataset for all videos and helmet masks from train or test directory."""
    def __init__(self, path_to_npy_files):
        self.video_files = os.listdir(path_to_npy_files)
        self.base_path = path_to_npy_files
        self.unique_ids = list(set(map(lambda s : s.split("_")[0] + "_" +
            s.split("_")[1], 
            self.video_files)))
        if "train" in path_to_npy_files:
            self.type = "train"
        else:
            self.type = "test"
        self.helmets_df = pd.read_csv(os.getcwd() + f"/nfl-player-contact-detection/{self.type}_baseline_helmets.csv")

    def __getitem__(self, idx):
        this_id  = self.unique_ids[idx]

        # Get video array all frame
        video = read_video(id=this_id, view="Sideline", type=self.type, cache=cache) # Returns np array
        return video   

    def __len__(self):
        return len(self.unique_ids)
    
def step_to_frame(step):
    return int(step/10*59.95+5*59.95)
    
def read_video(id, view, type, cache):
    """Reads video to numpy array using Open-CV"""
    
    filepath = f"nfl-player-contact-detection/{type}/{id}_{view}.mp4"
    if filepath not in cache:
        # Open the video file
        cap = cv2.VideoCapture(filepath)
        
        # Get video properties
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Initialize an empty numpy array to store the frames
        video_array = np.empty((num_frames, height, width), dtype=np.uint8)
        
        # Loop through each frame and store it in the numpy array
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            video_array[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            
        # Release the video object
        cap.release()
        
        # Update cache
        cache[filepath] = video_array
    
    return cache[filepath]

def view_contact(video_array, helmet_mask):
    """Plots contact with helmet mask."""
    plt.imshow(video_array)
    plt.imshow(helmet_mask, alpha=0.5, cmap='Reds')  
    plt.show()
    plt.close()

def create_boxes_dict(id, view, array_size, helmet_df, player_id, frames, cache):
    """Creates array of helmet mask boxes from helmet tracking data."""
    dict_key = f"{id}_{view}_{player_id}"
    if dict_key not in cache:
        cache[dict_key] = {}

    # Query the DataFrame once for all frames
    frame_data = helmet_df.query("view==@view & game_play==@id & nfl_player_id==@player_id")
    frame_data = frame_data.set_index('frame')

    frames_array = []
    for frame_idx in frames:
        if frame_idx not in cache[dict_key]:
            if frame_idx in frame_data.index:
                left, top, width, height = frame_data.loc[frame_idx, ['left', 'top', 'width', 'height']]

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
            cache[dict_key][frame_idx] = boxes_array
        frames_array.append(cache[dict_key][frame_idx])

    return np.stack(frames_array, axis=0)

class ContactDataset:
    """Dataset for training and testing zoomed contact examples."""

    # TO DO: Cross validate image size, add multiple frames per play, cross validate how many plays back forward, cross val skips

    def __init__(self, record_df_path, ground = False, feature_size=256, num_back_forward_steps=2, skips=1, distance_cutoff=5,
                num_per_classification=10000):
            
        self.ground=ground
        if ground:
            self.pos_class = pd.read_csv(record_df_path).query("nfl_player_id_2 == 'G' & contact == 1").reset_index(drop=1).sample(n=num_per_classification, replace=False, random_state=1)
            self.neg_class = pd.read_csv(record_df_path).query("nfl_player_id_2 == 'G' & contact == 0").reset_index(drop=1).sample(n=num_per_classification, replace=False, random_state=1)
        else:
            self.pos_class = pd.read_csv(record_df_path).query("nfl_player_id_2 != 'G'  & contact == 1").reset_index(drop=1).sample(n=num_per_classification, replace=False, random_state=1)
            self.neg_class = pd.read_csv(record_df_path).query("nfl_player_id_2 != 'G'  & contact == 0").reset_index(drop=1).sample(n=num_per_classification, replace=False, random_state=1)
        self.record_df = pd.concat([self.pos_class, self.neg_class], axis = 0)
        if "train" in record_df_path:
            self.type = "train"
        else:
            self.type = "test"
        print(f"Data Sample Contains {self.record_df.shape[0]} observations.")

        self.tracking_df = pd.read_csv(os.getcwd() + f"/nfl-player-contact-detection/{self.type}_player_tracking.csv")
        self.helmets_df = pd.read_csv(os.getcwd() + f"/nfl-player-contact-detection/{self.type}_baseline_helmets.csv")
        self.feature_size = feature_size
        self.skips=skips
        self.num_back_forward_steps=num_back_forward_steps
        self.distance_cutoff = distance_cutoff

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


    def __getitem__(self, idx):
        # Organize record info

        contact_info_df = self.record_df.iloc[idx,:]
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
            distance_as_mat = np.full((1, len(distance), self.feature_size, self.feature_size), distance[:, None, None])

        # Get video arrays and helmet masks
        video_arrays = []
        mask_arrays = []
        centerpoints = {}
        i=0
        for view in ["Sideline", "Endzone"]:
            # Video array
            raw_frames=read_video(id=game_play, view=view, type=self.type, cache=cache)[frame_ids, :, :]
            dim_1, dim_2=raw_frames.shape[1], raw_frames.shape[2]
            raw_frames=np.pad(raw_frames, pad_width=[(0, 0)] + [(half_feature_size, half_feature_size)] * 2, mode='constant', constant_values=0)
            
            # Helmet masks
            helmet_mask_player_1_dict = create_boxes_dict(id=game_play, view=view, array_size = (dim_1, dim_2),
                                                        helmet_df=self.helmets_df, player_id = player_1_id, frames = frame_ids, cache=helmet_cache)
            helmet_mask_player_2_dict = create_boxes_dict(id=game_play, view=view, array_size = (dim_1, dim_2),
                                                        helmet_df=self.helmets_df, player_id = player_2_id, frames = frame_ids, cache=helmet_cache)
            
            helmet_mask_frames = helmet_mask_player_1_dict + helmet_mask_player_2_dict
            helmet_mask_frames = np.pad(helmet_mask_frames, pad_width=[(0, 0)] + [(half_feature_size, half_feature_size)] * 2, mode='constant', constant_values=0)
            # Centerpoints & Zoom
            helmet_mask_df = self.helmets_df.query("view==@view & game_play==@game_play & frame==@frame_id")
            df_this_frame = helmet_mask_df.loc[helmet_mask_df['nfl_player_id'].isin([player_1_id, player_2_id])]
            if df_this_frame.empty:
                centerpoint = (raw_frames.shape[1] // 2, raw_frames.shape[2] // 2)
            else:
                x = np.mean(df_this_frame['left'].values + (df_this_frame['width'].values / 2))
                y = np.mean(df_this_frame['top'].values - (df_this_frame['height'].values / 2))
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

        return tuple(torch.Tensor(feature_array[:, timestep, :, :]) for timestep in range(feature_array.shape[1])), torch.Tensor([label])

    def __len__(self):
        return len(self.record_df)
