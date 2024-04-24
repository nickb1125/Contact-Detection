"""For removal of all frame backrounds in train and test sets"""

from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
from objects import preprocess_image, postprocess_image
import cv2
import os
from tqdm import tqdm
import numpy as np

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

# Get video save options

print("------Saving Backround Removed Copies---")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
for filepath in tqdm(all_file_paths):

    # Open the video file
    cap = cv2.VideoCapture(filepath)

    # Get video properties
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize an empty numpy array to store the frames
    video_array = np.empty((num_frames, height, width), dtype=np.uint8)

    # Get information for save
    base = filepath.split("/")[-1]
    if "train" in filepath:
        output_file_path = os.getcwd() + "/nfl-player-contact-detection/train/backround_removal/{base}.mp4"
    else:
        output_file_path = os.getcwd() + "/nfl-player-contact-detection/test/backround_removal/{base}.mp4"
    frame_size = (width, height)  # Use the same frame size as the original video
    video_writer = cv2.VideoWriter(output_file_path, fourcc, fps, frame_size)

    # Loop through each frame and store it in the numpy array
    for i in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        orig_im_size = frame.shape[0:2]
        image = preprocess_image(frame, orig_im_size).to(device)
        result=backround_removal_model(image)
        final = postprocess_image(result[0][0], orig_im_size)
        
        # Save frame
        video_writer.write(final)
        
    # Release the video object
    cap.release()

    # Release the VideoWriter object
    video_writer.release()
    
