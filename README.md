# Pose-Based Human Action Recognition

This is a lightweight pose-based action recognition pipeline built using PyTorch and MediaPipe.  
It uses 2D pose keypoints extracted from RGB videos to classify human actions via a deep recurrent LSTM model.

---

# Dataset

The dataset used in this project is a custom video-based dataset containing 7 human actions:

- Clapping  
- Meet and Split  
- Sitting  
- Standing Still  
- Walking  
- Walking While Reading Book  
- Walking While Using Phone

[Dataset Link](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset?select=Human+Activity+Recognition+-+Video+Dataset)

You can organize your dataset in the following format:
# Working Directory
      data/
      ├── Clapping/
      │ ├── clip1.mp4
      │ └── clip2.mp4
      ├── Sitting/
      │ ├── sit1.mp4
      │ └── sit2.mp4
      ...


---

# Working Directory

The working directory structure is organized as follows:

      pose-action-recognition/
      ├── data/ # Raw video files by action class
      ├── data_skeleton/ # Extracted pose sequences (.npy)
      ├── trial/ # New videos for inference
      ├── models/ # Saved LSTM checkpoints
      ├── src/
      │ ├── pose_extraction.py # Extract 33 pose keypoints using MediaPipe
      │ ├── skeleton_dataset.py # Load and preprocess fixed-length sequences
      │ ├── train_skeleton_lstm.py # Train LSTM model
      │ ├── evaluate_skeleton_lstm.py # Evaluate on test split with metrics
      │ ├── infer_skeleton.py # Inference on a single video
      │ └── batch_infer_skeleton.py # Batch inference on trial folder
      ├── requirements.txt
      └── README.md


---

# Requirements

All dependencies are listed in `requirements.txt`. Install using:

  pip install -r requirements.txt
  
  Main packages used:
  
    python==3.12.7
    pytorch==2.3.1
    numpy
    opencv-python
    mediapipe==0.10.9
    matplotlib
    scikit-learn
    tqdm

# Testing and Training

To test or train the LSTM model on the dataset:

-  To test or train the LSTM model on the dataset.
-  Download and organize the dataset under `data/`.
-  Run `pose_extraction.py` to convert videos to `.npy` pose sequences.
-  Run `train_skeleton_lstm.py` to train the LSTM model.
-  Run `evaluate_skeleton_lstm.py` to see classification report and confusion matrix.
-  Place new videos inside the `trial/` folder for inference.
-  Run `batch_infer_skeleton.py` to predict labels for all videos in `trial/`.

You can also test a single video like this:\
    python src/infer_skeleton.py path/to/your_clip.mp4

---

# Acknowledgment

This project was inspired by the work presented in **A Robust Framework for Abnormal Human Action Recognition Using R -Transform and Zernike Moments in Depth Videos** by  **Chhavi Dhiman** and **Dinesh Kumar Vishwakarma**. While the code has been entirely rewritten and adapted for new functionalities, their research provided valuable insights that helped shape this work.
I would also like to express my sincere gratitude to **Dr. Ashish Phophalia, Professor at IIIT Vadodara** for their guidance and support throughout this project.

You can find their paper here: [Link to the paper](https://ieeexplore.ieee.org/document/8662635)
