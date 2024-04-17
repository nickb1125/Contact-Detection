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

def create_boxes_array(frames_data, array_size):
    """Creates array of helmet mask boxes from helmet tracking data."""
    frames = list(set(frames_data.frame))
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
        print(idx)
        contact_info_df = self.record_df.iloc[idx,:].copy()
        label = int(contact_info_df.contact)
        game_play = contact_info_df.game_play
        player_1_id = int(contact_info_df.nfl_player_id_1)
        if not self.ground:
            player_2_id = int(contact_info_df.nfl_player_id_2)
        else:
            player_2_id = "G"
        step = contact_info_df.step
        frame_id = step_to_frame(step)
        steps = list(range(step-self.num_back_forward_steps*self.skips, 
                           step+self.num_back_forward_steps*self.skips+1, self.skips))
        frame_ids = sorted([step_to_frame(x) for x in steps])
        
        half_feature_size=int(self.feature_size/2)

        # Get distance info (if not ground play)
        if not self.ground:
            p1_row_track = self.tracking_df.query("game_play==@game_play & (step in @steps) & nfl_player_id==@player_1_id")
            p2_row_track = self.tracking_df.query("game_play==@game_play & (step in @steps) & nfl_player_id==@player_2_id")
            distance = np.sqrt((p1_row_track.x_position.values - p2_row_track.x_position.values)**2 + 
                            (p1_row_track.y_position.values - p2_row_track.y_position.values)**2)
            distance_as_mat = np.stack([np.full((self.feature_size, self.feature_size), x) for x in distance])
        
        # Get video array sideline
        video_array = read_video(id=contact_info_df.game_play, 
                          view="Sideline", 
                          type=self.type) 
        image_side = np.mean(video_array[frame_ids,:,:,:], axis= 3)

        # Get video array endzone
        video_array = read_video(id=contact_info_df.game_play, 
                          view="Endzone", 
                          type=self.type) 
        image_end = np.mean(video_array[frame_ids,:,:,:], axis= 3)

        # Get pixel centerpoint of contact sideline
        helmet_mask_df = self.helmets_df.query("view=='Sideline' & game_play==@game_play & (frame in @frame_ids)")
        helmet_mask_df_side = helmet_mask_df.loc[helmet_mask_df.nfl_player_id.isin([player_1_id, player_2_id])]
        if set(frame_ids) != set(helmet_mask_df_side.frame):
            raise IndexError("Missing frames from steps specified. Try reducing step size if possible.")
        if player_1_id not in [int(x) for x in helmet_mask_df_side.nfl_player_id]:
            print(idx)
            print("-----")
            print("Player 1 not in side view.")
        if player_2_id not in [int(x) for x in helmet_mask_df_side.nfl_player_id]:
            print(idx)
            print("-----")
            print("Player 2 not in side view.")
        if helmet_mask_df_side.empty:
            print(idx)
            print("-----")
            print("Neither player in side view.")
            centerpoint_side = [(image_side.shape[0]/2, image_side.shape[1]/2)] * (self.num_back_forward_steps*2+1)
        else:
            df_this_frame=helmet_mask_df_side.loc[helmet_mask_df_side.frame==frame_id]
            x, y = np.mean(df_this_frame.left.values+(df_this_frame.width.values/2)), np.mean(df_this_frame.top.values-(df_this_frame.height.values/2))
            centerpoint_side=(int(x) + half_feature_size, int(y) + half_feature_size)
    
        # Get pixel centerpoint of contact endzone
        helmet_mask_df = self.helmets_df.query("view=='Endzone' & game_play==@game_play & (frame in @frame_ids)")
        helmet_mask_df_end = helmet_mask_df.loc[helmet_mask_df.nfl_player_id.isin([player_1_id, player_2_id])]
        print(helmet_mask_df_end)
        if set(frame_ids) != set(helmet_mask_df_end.frame):
            raise IndexError("Missing frames from steps specified. Try reducing step size if possible.")
        if player_1_id not in [int(x) for x in helmet_mask_df_end.nfl_player_id]:
            print(idx)
            print("-----")
            print("Player 1 not in end view.")
        if player_2_id not in [int(x) for x in helmet_mask_df_end.nfl_player_id]:
            print(idx)
            print("-----")
            print("Player 2 not in end view.")
        if helmet_mask_df_end.empty:
            print(idx)
            print("-----")
            print("Neither player in end view")
            centerpoint_end = [(int(image_end.shape[1]/2), int(image_end.shape[2]/2))] * (self.num_back_forward_steps*2+1)
        else:
            df_this_frame=helmet_mask_df_end.loc[helmet_mask_df_end.frame==frame_id]
            x, y = np.mean(df_this_frame.left.values+(df_this_frame.width.values/2)), np.mean(df_this_frame.top.values-(df_this_frame.height.values/2))
            centerpoint_end=(int(x) + half_feature_size, int(y) + half_feature_size)

        # Get helmet mask
        if helmet_mask_df_side.empty:
            helmet_mask_frame_side = np.zeros((len(frame_ids), image_side.shape[1], image_side.shape[2]))
        else:
            helmet_mask_frame_side = create_boxes_array(helmet_mask_df_side, (image_side.shape[1], image_side.shape[2]))
        if helmet_mask_df_end.empty:
            helmet_mask_frame_end = np.zeros((len(frame_ids), image_end.shape[1], image_end.shape[2]))
        else:
            helmet_mask_frame_end = create_boxes_array(helmet_mask_df_end, (image_end.shape[1], image_end.shape[2]))

        # Pad images in case of zoom out of bounds
        pad_width = [(0, 0)] + [(half_feature_size, half_feature_size) for _ in range(2)]
        image_side=np.pad(image_side, pad_width=pad_width, mode='constant', constant_values=0)
        image_end=np.pad(image_end, pad_width=pad_width, mode='constant', constant_values=0)
        helmet_mask_frame_side= np.pad(helmet_mask_frame_side, pad_width=pad_width, mode='constant', constant_values=0)
        helmet_mask_frame_end= np.pad(helmet_mask_frame_end, pad_width=pad_width, mode='constant', constant_values=0)

        print(centerpoint_side)
        print(centerpoint_end)

        if self.ground:
            ret = tuple(np.stack((
                image_side[timestep, (centerpoint_side[1]-half_feature_size):(centerpoint_side[1]+half_feature_size), 
                        (centerpoint_side[0]-half_feature_size):(centerpoint_side[0]+half_feature_size)], 
                helmet_mask_frame_side[timestep, (centerpoint_side[1]-half_feature_size):(centerpoint_side[1]+half_feature_size), 
                                    (centerpoint_side[0]-half_feature_size):(centerpoint_side[0]+half_feature_size)],
                image_end[timestep, (centerpoint_end[1]-half_feature_size):(centerpoint_end[1]+half_feature_size), 
                        (centerpoint_end[0]-half_feature_size):(centerpoint_end[0]+half_feature_size)], 
                helmet_mask_frame_end[timestep, (centerpoint_end[1]-half_feature_size):(centerpoint_end[1]+half_feature_size), 
                                    (centerpoint_end[0]-half_feature_size):(centerpoint_end[0]+half_feature_size)]), 
                                    axis = 2) for timestep in range(len(frame_ids)))
        else:
            ret = tuple(np.stack((
                image_side[timestep, (centerpoint_side[1]-half_feature_size):(centerpoint_side[1]+half_feature_size), 
                        (centerpoint_side[0]-half_feature_size):(centerpoint_side[0]+half_feature_size)], 
                helmet_mask_frame_side[timestep, (centerpoint_side[1]-half_feature_size):(centerpoint_side[1]+half_feature_size), 
                                    (centerpoint_side[0]-half_feature_size):(centerpoint_side[0]+half_feature_size)],
                image_end[timestep, (centerpoint_end[1]-half_feature_size):(centerpoint_end[1]+half_feature_size), 
                        (centerpoint_end[0]-half_feature_size):(centerpoint_end[0]+half_feature_size)], 
                helmet_mask_frame_end[timestep, (centerpoint_end[1]-half_feature_size):(centerpoint_end[1]+half_feature_size), 
                                    (centerpoint_end[0]-half_feature_size):(centerpoint_end[0]+half_feature_size)],
                distance_as_mat[timestep, :, :]), axis = 2) for timestep in range(len(frame_ids)))

        return tuple(torch.Tensor(feature) for feature in ret), torch.Tensor(label)

    def __len__(self):
        return len(self.record_df)