#!/usr/bin/env python
"""
File: batch_infer_skeleton.py
Purpose: Batch‐infer actions on all videos in the `trial/` folder using the trained skeleton‐LSTM model.
"""

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from tqdm import tqdm

from train_skeleton_lstm import SkeletonLSTM, classes, seq_len, hidden_size, num_layers

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

def extract_frames(video_path):
    """
    Read a video file and return a list of BGR frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_keypoints(frames):
    """
    Given a list of BGR frames, extract normalized (x,y) for 33 landmarks each.
    Returns an np.array of shape (T,33,2).
    """
    keypoints = []
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        for frame in frames:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            if res.pose_landmarks:
                pts = [[lm.x, lm.y] for lm in res.pose_landmarks.landmark]
            else:
                pts = [[0.0, 0.0]] * 33
            keypoints.append(pts)
    return np.array(keypoints, dtype=np.float32)

def pad_or_truncate(kp):
    """
    Pad with zeros or truncate the keypoint sequence to length `seq_len`.
    Returns an array of shape (seq_len,33,2).
    """
    T = kp.shape[0]
    if T < seq_len:
        pad = np.zeros((seq_len - T, 33, 2), dtype=np.float32)
        kp = np.concatenate([kp, pad], axis=0)
    else:
        kp = kp[:seq_len]
    return kp

def load_model(device):
    """
    Instantiate the LSTM model and load the best checkpoint.
    """
    model = SkeletonLSTM(input_size=33*2,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         num_classes=len(classes)).to(device)
    script_dir = os.path.dirname(__file__)
    ckpt = os.path.join(script_dir, 'models', 'best_skeleton_lstm.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model

def predict_video(video_path, model, device):
    """
    Run one video through the pipeline and return (predicted_class_index, confidence).
    """
    frames = extract_frames(video_path)
    if not frames:
        return None, None
    kp = extract_keypoints(frames)                     # (T,33,2)
    kp = pad_or_truncate(kp)                           # (seq_len,33,2)
    inp = kp.reshape(1, seq_len, -1)                   # (1,seq_len,66)
    x = torch.from_numpy(inp).to(device)
    with torch.no_grad():
        logits = model(x)                              # (1,num_classes)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()

def main():
    # Resolve paths
    script_dir = os.path.dirname(__file__)
    trial_dir  = os.path.normpath(os.path.join(script_dir, '..', 'trial'))
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(trial_dir):
        print(f"Error: trial folder not found at {trial_dir}")
        return

    model = load_model(device)

    # Find all video files in trial/
    patterns = ['*.mp4', '*.avi', '*.mov']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(trial_dir, p)))
    if not files:
        print(f"No videos found in {trial_dir}")
        return

    # Batch inference
    print(f"Running inference on {len(files)} videos in {trial_dir}")
    for vid in tqdm(sorted(files), desc="Batch infer"):
        pred_idx, conf = predict_video(vid, model, device)
        if pred_idx is None:
            print(f"[ERROR] Could not read frames from {os.path.basename(vid)}")
        else:
            print(f"{os.path.basename(vid):<30} -> {classes[pred_idx]:<25} ({conf*100:.1f}%)")

if __name__ == '__main__':
    main()
