""" new iteration """

"""
File: evaluate_skeleton_lstm.py
Purpose: Loads the single, fixed test split and the BEST checkpoint,
         then prints per‐class metrics and a confusion matrix.
"""

import os
import sys
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from skeleton_dataset import SkeletonDataset
from train_skeleton_lstm import SkeletonLSTM, classes, seq_len, hidden_size, num_layers
matplotlib.use('TkAgg')

def load_model(path, device):
    model = SkeletonLSTM(33 * 2, hidden_size, num_layers, len(classes)).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate():
    # resolve paths
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data_skeleton'))
    splits_dir = os.path.join(script_dir, 'splits')
    model_path = os.path.join(script_dir, 'models', 'best_skeleton_lstm.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load full dataset & test indices
    full_ds = SkeletonDataset(data_dir, classes, seq_len=seq_len)
    test_idx = np.load(os.path.join(splits_dir, 'test_idx.npy'))
    test_ds = Subset(full_ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f" [Eval] {len(test_ds)} samples in test split")
    if not os.path.isfile(model_path):
        print("Error: best checkpoint not found:", model_path)
        sys.exit(1)

    # load model
    model = load_model(model_path, device)

    # run inference
    all_true, all_pred = [], []
    for seqs, labels in tqdm(test_loader, desc="Evaluating"):
        seqs, labels = seqs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(seqs).argmax(dim=1)
        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # metrics
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    evaluate()
