import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch

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
        self.helmets_df = pd.read_csv(f"/Users/nickbachelder/Desktop/Personal Code/Kaggle/Contact/nfl-player-contact-detection/{self.type}_baseline_helmets.csv")

    def __getitem__(self, idx):
        this_id  = self.unique_ids[idx]

        # Get video array all frame
        video = read_video(id=this_id, view="Sideline", type=self.type) # Returns np array
        return video   

    def __len__(self):
        return len(self.unique_ids)
    
def step_to_frame(step):
    return int(step/10*59.95+5*59.95)
    
def read_video(id, view, type):
    """Reads video to numpy array using Open-CV"""
    # Open the video file
    filepath = f"nfl-player-contact-detection/{type}/{id}_{view}.mp4"
    cap = cv2.VideoCapture(filepath)
    
    # Get video properties
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Initialize an empty numpy array to store the frames
    video_array = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    
    # Loop through each frame and store it in the numpy array
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        video_array[i] = frame
        
    # Release the video object
    cap.release()
    
    return video_array
    
def view_video(video_array):
    """Plots series of frame examples with helmet maskes from play."""
    num_frames = video_array.shape[0]
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    spacing = int(num_frames / 15)

    frame_count = 0
    for i in range(2):
        for j in range(5):
            axes[i, j].imshow(cv2.cvtColor(video_array[frame_count], cv2.COLOR_BGR2RGB))
            # Plot mask image
            axes[i, j].axis('off')
            frame_count += spacing
    plt.show()

def view_contact(video_array, helmet_mask):
    """Plots contact with helmet mask."""
    plt.imshow(video_array)
    plt.imshow(helmet_mask, alpha=0.5, cmap='Reds')  
    plt.show()
    plt.close()

def create_boxes_array(frames_data, array_size, frames):
    """Creates array of helmet mask boxes from helmet tracking data."""
    boxes_arrays = []

    for frame_idx in frames:
        frame_data = frames_data.loc[frames_data.frame == frame_idx][['left', 'top', 'width', 'height']].values
        # Initialize an empty numpy array for the current frame
        boxes_array = np.zeros(array_size, dtype=np.uint8)
        for box in frame_data:
            left, top, width, height = box

            # Ensure box coordinates are within the array bounds
            left = max(0, left)
            top = max(0, top)
            right = min(array_size[1], left + width)
            bottom = min(array_size[0], top + height)

            # Mark the region of the box in the numpy array
            boxes_array[top:bottom, left:right] = 255

        # Append the numpy array for the current frame to the list
        boxes_arrays.append(boxes_array)

    return np.array(boxes_arrays)

class ContactDataset:
    """Dataset for training and testing zoomed contact examples."""

    # TO DO: Cross validate image size, add multiple frames per play, cross validate how many plays back forward, cross val skips

    def __init__(self, record_df_path, ground = False, feature_size=256, num_back_forward_steps=2, skips=1):
        self.ground=ground
        if ground:
            self.record_df = pd.read_csv(record_df_path).query("nfl_player_id_2 == 'G'").reset_index(drop=1)
        else:
            self.record_df = pd.read_csv(record_df_path).query("nfl_player_id_2 != 'G'").reset_index(drop=1)
        if "train" in record_df_path:
            self.type = "train"
        else:
            self.type = "test"
        self.tracking_df = pd.read_csv(f"/Users/nickbachelder/Desktop/Personal Code/Kaggle/Contact/nfl-player-contact-detection/{self.type}_player_tracking.csv")
        self.helmets_df = pd.read_csv(f"/Users/nickbachelder/Desktop/Personal Code/Kaggle/Contact/nfl-player-contact-detection/{self.type}_baseline_helmets.csv")
        self.feature_size = feature_size
        self.skips=skips
        self.num_back_forward_steps=num_back_forward_steps

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
            p1_row_track = self.tracking_df.query("game_play==@game_play & (step in @steps) & nfl_player_id==@player_1_id")
            p2_row_track = self.tracking_df.query("game_play==@game_play & (step in @steps) & nfl_player_id==@player_2_id")
            distance = np.sqrt((p1_row_track['x_position'].values - p2_row_track['x_position'].values)**2 + 
                            (p1_row_track['y_position'].values - p2_row_track['y_position'].values)**2)
            distance_as_mat = np.stack([np.full((self.feature_size, self.feature_size), x) for x in distance])

        # Get video arrays
        video_arrays = {}
        for view in ["Sideline", "Endzone"]:
            video_array = read_video(id=game_play, view=view, type=self.type)
            video_arrays[view] = np.mean(video_array[frame_ids,:,:,:], axis=3)

        # Get pixel centerpoint of contact for each view
        centerpoints = {}
        for view in ["Sideline", "Endzone"]:
            helmet_mask_df = self.helmets_df.query("view==@view & game_play==@game_play & (frame in @frame_ids)")
            helmet_mask_df_view = helmet_mask_df.loc[helmet_mask_df['nfl_player_id'].isin([player_1_id, player_2_id])]
            if helmet_mask_df_view.empty:
                centerpoints[view] = (video_arrays[view].shape[1] // 2, video_arrays[view].shape[2] // 2)
            else:
                df_this_frame = helmet_mask_df_view.loc[helmet_mask_df_view['frame'] == frame_id]
                x = np.mean(df_this_frame['left'].values + (df_this_frame['width'].values / 2))
                y = np.mean(df_this_frame['top'].values - (df_this_frame['height'].values / 2))
                centerpoints[view] = (int(x) + half_feature_size, int(y) + half_feature_size)
        
        ret = []
        for timestep in range(len(frame_ids)):
            timestep_ret = []
            for view in ["Sideline", "Endzone"]:
                image = np.pad(video_arrays[view], pad_width=[(0, 0)] + [(half_feature_size, half_feature_size)] * 2, mode='constant', constant_values=0)
                helmet_mask_df = self.helmets_df.query("view==@view & game_play==@game_play & (frame in @frame_ids)")
                helmet_mask_df_view = helmet_mask_df.loc[helmet_mask_df['nfl_player_id'].isin([player_1_id, player_2_id])]
                helmet_mask_frame = create_boxes_array(helmet_mask_df_view, (image.shape[1], image.shape[2]), frames=frame_ids)
                helmet_mask_frame = np.pad(helmet_mask_frame, pad_width=[(0, 0)] + [(half_feature_size, half_feature_size)] * 2, mode='constant', constant_values=0)
                feature = [image[timestep,
                                        (centerpoints[view][1]-half_feature_size):(centerpoints[view][1]+half_feature_size), 
                                        (centerpoints[view][0]-half_feature_size):(centerpoints[view][0]+half_feature_size)],
                           helmet_mask_frame[timestep,
                                        (centerpoints[view][1]-half_feature_size):(centerpoints[view][1]+half_feature_size), 
                                        (centerpoints[view][0]-half_feature_size):(centerpoints[view][0]+half_feature_size)]]
                timestep_ret.extend(feature)
            if not self.ground:
                timestep_ret.append(distance_as_mat[timestep, :, :])
            ret.append(np.stack(timestep_ret, axis = 2))

        return tuple(torch.Tensor(feature) for feature in ret), torch.Tensor(label)

    def __len__(self):
        return len(self.record_df)