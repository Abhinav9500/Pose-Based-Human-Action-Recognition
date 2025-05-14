"""
File: skeleton_dataset.py
Purpose: Defines a PyTorch Dataset to load skeleton .npy files and their labels
         for LSTM training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, skeleton_dir, classes, seq_len=100):
        self.samples = []  # list of (path, label)
        self.classes = classes
        self.seq_len = seq_len
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(skeleton_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, fname),
                                         self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path)  # shape (T, 33, 2)
        T = seq.shape[0]
        # pad or truncate to seq_len
        if T < self.seq_len:
            pad = np.zeros((self.seq_len - T, 33, 2), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:self.seq_len]
        # flatten joints: (seq_len, 66)
        seq = seq.reshape(self.seq_len, -1)
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)
