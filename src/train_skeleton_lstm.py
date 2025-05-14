"""
File: train_skeleton_lstm.py
Purpose: Splits data once (80/20), trains on skeleton .npy files with an LSTM,
         and saves both the best‐on‐val and final checkpoints.
"""

import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from skeleton_dataset import SkeletonDataset

# ── Hyperparameters & seeds ──────────────────────────────────────────────────
epochs = 500
batch_size = 32
learning_rate = 1e-3
seq_len = 100
hidden_size = 128
num_layers = 2
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ── Resolve paths ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(__file__)
data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data_skeleton'))
models_dir = os.path.join(script_dir, 'models')
splits_dir = os.path.join(script_dir, 'splits')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

# ── Discover classes ─────────────────────────────────────────────────────────
classes = sorted(
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
)
print(f"Using classes: {classes}")

# ── Build dataset & indices ─────────────────────────────────────────────────
dataset = SkeletonDataset(data_dir, classes, seq_len=seq_len)
N = len(dataset)
print(f"Found {N} samples in {data_dir}")
if N == 0:
    print("Error: no .npy skeleton files found. Run pose_extraction.py first.")
    sys.exit(1)

# labels list for stratification
labels = [dataset.samples[i][1] for i in range(N)]

# single, fixed 80/20 split: train_val vs test
idx = list(range(N))
train_val_idx, test_idx = train_test_split(
    idx, test_size=0.20, random_state=seed, stratify=labels
)

# within train_val, do a 75/25 split → 60/20/20 overall
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.25,
    random_state=seed,
    stratify=[labels[i] for i in train_val_idx]
)

# save the splits once
np.save(os.path.join(splits_dir, 'train_idx.npy'), train_idx)
np.save(os.path.join(splits_dir, 'val_idx.npy'), val_idx)
np.save(os.path.join(splits_dir, 'test_idx.npy'), test_idx)
print(f"Saved splits in {splits_dir} (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

# create subsets
train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)


# ── Model Definition ─────────────────────────────────────────────────────────
class SkeletonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time-step
        return self.fc(out)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkeletonLSTM(input_size=33 * 2,
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_acc = 0.0
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}", end='  ')

    # validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            preds = model(seqs).argmax(dim=1)
            correct += (preds == labels).sum().item()
    acc = correct / len(val_ds) * 100
    print(f"Val Acc: {acc:.2f}%")

    # checkpoint best
    if acc > best_acc:
        best_acc = acc
        best_path = os.path.join(models_dir, 'best_skeleton_lstm.pth')
        torch.save(model.state_dict(), best_path)
        print(f" → Saved BEST (Acc={acc:.2f}%) to {best_path}")

# final checkpoint
final_path = os.path.join(models_dir, 'skeleton_lstm.pth')
torch.save(model.state_dict(), final_path)
print(f"Saved final model to {final_path}")
