"""File: pose_extraction.py
Purpose: Extract 2D pose keypoints from each video using MediaPipe, and save
         skeleton sequences as .npy files for downstream ST‑GCN or LSTM training.
"""

import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose


def extract_frames(video_path):
    """
    Load a video file and return a list of BGR frames.
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
    keypoints = []
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        for frame in frames:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            if results.pose_landmarks:
                pts = []
                for lm in results.pose_landmarks.landmark:
                    pts.append([lm.x, lm.y])
                keypoints.append(pts)
            else:
                # No detection: zero fill
                keypoints.append([[0.0, 0.0]] * 33)
    return np.array(keypoints, dtype=np.float32)  # (T, 33, 2)


def process_dataset(videos_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for action in os.listdir(videos_dir):
        src_action = os.path.join(videos_dir, action)
        dst_action = os.path.join(out_dir, action)
        if not os.path.isdir(src_action):
            continue
        os.makedirs(dst_action, exist_ok=True)

        for vid in os.listdir(src_action):
            if not vid.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
            src_path = os.path.join(src_action, vid)
            base, _ = os.path.splitext(vid)
            dst_path = os.path.join(dst_action, base + '.npy')
            # Skip if already processed
            if os.path.exists(dst_path):
                continue

            frames = extract_frames(src_path)
            if len(frames) == 0:
                continue
            keypoints = extract_keypoints(frames)
            np.save(dst_path, keypoints)
            print(f"Saved skeletons for {src_path} → {dst_path}")


if __name__ == '__main__':
    videos_dir = '../data'  # Your original videos directory
    out_dir = '../data_skeleton'  # Where to save the .npy skeleton files
    process_dataset(videos_dir, out_dir)
