"""
File: infer_skeleton.py
Purpose: Run a single video through MediaPipe + skeleton LSTM and print the predicted action,
         with a tqdm progress bar over frames when extracting keypoints.
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm
from train_skeleton_lstm import SkeletonLSTM, classes, seq_len, hidden_size, num_layers

# Initialize MediaPipe Pose once
mp_pose = mp.solutions.pose


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_keypoints_with_progress(frames):
    keypoints = []
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        for frame in tqdm(frames, desc="Extracting keypoints"):
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            if results.pose_landmarks:
                pts = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
            else:
                pts = [[0.0, 0.0]] * 33
            keypoints.append(pts)
    return np.array(keypoints, dtype=np.float32)


def predict(video_path, model_path='models/skeleton_lstm.pth'):
    # 1) Extract frames
    frames = extract_frames(video_path)
    if not frames:
        print("Error: no frames extracted.")
        return

    # 2) Extract keypoints with progress
    keypoints = extract_keypoints_with_progress(frames)  # (T,33,2)
    T = keypoints.shape[0]

    # 3) Pad or truncate to seq_len
    if T < seq_len:
        pad = np.zeros((seq_len - T, 33, 2), dtype=np.float32)
        keypoints = np.concatenate([keypoints, pad], axis=0)
    else:
        keypoints = keypoints[:seq_len]

    # 4) Prepare input tensor
    inp = keypoints.reshape(1, seq_len, -1)  # (1, seq_len, 66)
    x = torch.from_numpy(inp).to(
        next(torch.cuda.device_count() and [torch.device('cuda')])[0] if torch.cuda.is_available() else torch.device(
            'cpu'))

    # 5) Load model
    device = x.device
    model = SkeletonLSTM(input_size=33 * 2,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 6) Predict
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    print(f"Predicted action: {classes[pred]}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python infer_skeleton.py <path_to_video>")
        sys.exit(1)
    video_file = sys.argv[1]
    if not os.path.isfile(video_file):
        print(f"Error: File not found: {video_file}")
        sys.exit(1)
    predict(video_file)
