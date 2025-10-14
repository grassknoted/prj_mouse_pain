import os
import glob
import pandas as pd
import numpy as np
import cv2

VIDEOS_DIR = "VIDEOS/"
ANNOTATIONS_DIR = "PROCESSED_ANNOTATIONS/"
DLC_FILE_EXT = "DLC_resnet50_pawtracking_comparisionMay7shuffle1_500000.csv"
OUTPUT_X_FILE = "X_ordered_processed.npy"
OUTPUT_Y_FILE = "y_ordered_processed.npy"
OUTPUT_FRAMES_FILE = "X_frames_processed.npy"

frame_rate = 30
trial_length_seconds = 12
frames_per_trial = trial_length_seconds * frame_rate
likelihood_threshold = 0.6
frame_shape = (128, 128)  # Resize frames to a fixed size

def load_pose_data(pose_file):
    # Load the file with three header rows
    pose_df = pd.read_csv(pose_file, header=[0, 1, 2])

    # Flatten the multi-level columns into single-level columns
    pose_df.columns = [
        f"{col[1]}_{col[2]}" if col[2] else col[1] for col in pose_df.columns
    ]

    # Ensure 'Frame' column exists
    if 'Frame' not in pose_df.columns:
        pose_df['Frame'] = range(len(pose_df))

    return pose_df

def preprocess_trial_data(merged_df):
    """
    Preprocesses trial data to extract features and actions in the correct order.

    Args:
        merged_df (pd.DataFrame): Merged data containing pose and annotations.

    Returns:
        tuple: Features and actions for the trial.
    """
    ordered_body_parts = [
        "tail_base", "L_hindpaw", "L_frontpaw", "mouth", "R_frontpaw", "R_hindpaw"
    ]
    x_cols = [f"{part}_x" for part in ordered_body_parts]
    y_cols = [f"{part}_y" for part in ordered_body_parts]
    likelihood_cols = [f"{part}_likelihood" for part in ordered_body_parts]

    # Check if columns exist and fill missing ones with NaN
    for col in x_cols + y_cols + likelihood_cols:
        if col not in merged_df.columns:
            merged_df[col] = np.nan

    # Ensure all columns are numeric
    for col in x_cols + y_cols + likelihood_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # Extract positions and create feature array
    x_positions = merged_df[x_cols].values
    y_positions = merged_df[y_cols].values

    num_frames = len(merged_df)
    num_keypoints = len(ordered_body_parts)
    features = np.empty((num_frames, num_keypoints * 2), dtype=float)
    for i in range(num_keypoints):
        features[:, 2 * i] = x_positions[:, i]
        features[:, 2 * i + 1] = y_positions[:, i]

    # Extract actions
    actions = merged_df['Action'].values
    return features, actions

def extract_frames(video_path, trial_start_frame, frames_per_trial):
    """
    Extracts frames for a trial from the video.

    Args:
        video_path (str): Path to the video file.
        trial_start_frame (int): Starting frame of the trial.
        frames_per_trial (int): Number of frames to extract.

    Returns:
        list: List of processed frames as NumPy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, trial_start_frame)

    for _ in range(frames_per_trial):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, frame_shape)
        frame = frame / 255.0  # Normalize pixel values to [0, 1]
        frames.append(frame)

    cap.release()
    return frames

pose_data_cache = {}
all_trials_features = []
all_trials_actions = []
all_trials_frames = []

annotation_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*_*.csv"))
annotation_files.sort()

file_counter = 0

for ann_file in annotation_files:
    base_name = os.path.basename(ann_file)
    video_name_with_ext, trial_str = VIDEOS_DIR + "_".join(base_name.split('_')[1:-1]), base_name.split('_')[-1]
    trial_number = os.path.splitext(trial_str)[0]
    video_name = video_name_with_ext

    print(f"Video: {video_name}, Trial: {trial_number}")

    # Load the annotation for this trial
    ann_df = pd.read_csv(ann_file)
    dlc_file_name = video_name.replace(".mp4", DLC_FILE_EXT)

    try:
        # Load the pose data for the corresponding video
        pose_df = load_pose_data(dlc_file_name)
        trial_start_frame = (int(trial_number) - 1) * frames_per_trial
        ann_df['Frame'] = ann_df['Frame'] + trial_start_frame

        if dlc_file_name not in pose_data_cache:
            pose_data_cache[dlc_file_name] = load_pose_data(dlc_file_name)
        pose_df = pose_data_cache[dlc_file_name]

        merged_df = pd.merge(pose_df, ann_df, on='Frame', how='inner')

        if len(merged_df) != frames_per_trial:
            print(f"Warning: {ann_file} expected {frames_per_trial} frames, got {len(merged_df)} frames.")
            continue

        trial_features, trial_actions = preprocess_trial_data(merged_df)

        # Extract corresponding frames
        trial_frames = extract_frames(video_name, trial_start_frame, frames_per_trial)

        file_counter += 1
        print(f"Processed {file_counter}/{len(annotation_files)} {ann_file}")
        all_trials_features.append(trial_features)
        all_trials_actions.append(trial_actions)
        all_trials_frames.append(np.array(trial_frames, dtype=np.float32))
    except Exception as e:
        print(f"Error processing: {ann_file}, Error: {e}")
        continue

X = np.array(all_trials_features, dtype=float)
y = np.array(all_trials_actions, dtype=int)
frames = np.array(all_trials_frames, dtype=object)

np.save(OUTPUT_X_FILE, X)
np.save(OUTPUT_Y_FILE, y)
np.save(OUTPUT_FRAMES_FILE, frames)

print(f"Preprocessing complete. Processed {X.shape[0]} trials.")
print(f"X shape: {X.shape}, y shape: {y.shape}, frames shape: {frames.shape}")
