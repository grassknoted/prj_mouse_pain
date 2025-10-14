import os
import re
import pandas as pd

##############################################################################
# 1. Configuration: Folder paths and file naming patterns
##############################################################################

# Update these folder paths to match your local setup
ANNOTATIONS_FOLDER = "./REMY2/ALL_ANNOTATIONS"
DEEPLABCUT_FOLDER = "./REMY2/ALL_VIDEOS"

# We'll store the final combined CSV here
OUTPUT_CSV = "new_thesis_data.csv"

##############################################################################
# 2. Preprocessing Functions
##############################################################################

def parse_annotation_filename(filename):
    """
    Matches filenames like:
      prediction_<VIDEO_NAME>.mp4_<TRIAL_NUM>.csv
      OR
      predictions_<VIDEO_NAME>.mp4_<TRIAL_NUM>.csv

    <TRIAL_NUM> must be digits only. Excludes files with extra underscores.
    Returns (video_name_with_ext, trial_num) if valid, else (None, None).

    Example:
      'predictions_MyVideo.mp4_2.csv'
        -> ("MyVideo.mp4", 2)
    """
    pattern = re.compile(r"^predictions?_(.+\.mp4)_(\d+)\.csv$")
    match = pattern.match(filename)
    if not match:
        return None, None
    video_name_with_ext = match.group(1)  # includes .mp4
    trial_num = int(match.group(2))
    return video_name_with_ext, trial_num

def build_dlc_filename(video_name_with_ext):
    """
    Given the video name with extension (e.g., 'MyVideo.mp4'),
    remove '.mp4' and append 'DLC_resnet50_pawtracking_comparisionMay7shuffle1_500000.h5'.

    Example:
      'MyVideo.mp4' -> 'MyVideoDLC_resnet50_pawtracking_comparisionMay7shuffle1_500000.h5'
    """
    if video_name_with_ext.endswith(".mp4"):
        base_name = video_name_with_ext[:-4]  # strip off ".mp4"
    else:
        base_name = video_name_with_ext
    dlc_filename = f"{base_name}DLC_resnet50_pawtracking_comparisionMay7shuffle1_500000.h5"
    return dlc_filename

def load_annotation_csv(csv_path):
    """
    Loads an annotation CSV containing columns: [Frame, Action].
    We now expect exactly 360 rows (frames 0..359).
    Returns a DataFrame if it has exactly 360 rows, else None.
    """
    df = pd.read_csv(csv_path)
    if len(df) != 360:
        print(f"Warning: {csv_path} has {len(df)} rows, expected 360. Skipping.")
        return None
    required_cols = {"Frame", "Action"}
    if not required_cols.issubset(df.columns):
        print(f"Warning: {csv_path} missing required columns {required_cols}. Skipping.")
        return None
    return df

def load_dlc_h5(dlc_path):
    """
    Loads the DeepLabCut .h5 file. We want exactly 360 frames total (0..359).
    - If the file has < 360 frames, skip.
    - If it has more, we'll just slice the first 360 rows.

    Returns a DataFrame with shape (360, ...) if valid, else None.
    """
    try:
        df = pd.read_hdf(dlc_path)
    except Exception as e:
        print(f"Error reading H5 file {dlc_path}: {e}")
        return None

    if len(df) < 360:
        print(f"Warning: {dlc_path} has {len(df)} frames, fewer than 360. Skipping.")
        return None

    # Slice frames 0..359 (exactly 360 frames)
    df = df.iloc[:360].copy()

    if len(df) != 360:
        print(f"Warning: after slicing, {dlc_path} has {len(df)} rows, expected 360. Skipping.")
        return None

    return df

def preprocess_data():
    """
    Main function that:
    - Finds valid annotation files in ANNOTATIONS_FOLDER
    - For each, builds the DLC .h5 filename, checks existence
    - Loads & merges the 360-row annotation + DLC data
    - Returns a list of dicts with metadata and the merged DataFrame
    """
    all_trials_data = []

    # 1. Gather valid annotation files
    valid_files = []
    for fname in os.listdir(ANNOTATIONS_FOLDER):
        if not fname.endswith(".csv"):
            continue
        video_name_with_ext, trial_num = parse_annotation_filename(fname)
        if video_name_with_ext is not None:
            valid_files.append(fname)
        else:
            # Doesn't match the naming convention => exclude
            pass

    # 2. Process each valid annotation file
    for fname in valid_files:
        annotation_path = os.path.join(ANNOTATIONS_FOLDER, fname)
        video_name_with_ext, trial_num = parse_annotation_filename(fname)
        
        # Build the DLC filename
        dlc_filename = build_dlc_filename(video_name_with_ext)
        dlc_path = os.path.join(DEEPLABCUT_FOLDER, dlc_filename)

        # Check if DLC file exists
        if not os.path.exists(dlc_path):
            print(f"No DLC file found for {fname}. Expected: {dlc_filename}")
            continue

        # Load annotation (CSV)
        ann_df = load_annotation_csv(annotation_path)
        if ann_df is None:
            # Missing columns or not 360 frames
            continue

        # Load DLC data (H5)
        dlc_df = load_dlc_h5(dlc_path)
        if dlc_df is None or len(dlc_df) != 360:
            # DLC couldn't be read or doesn't have exactly 360 frames
            continue

        # Merge row-by-row (both DataFrames are 360 rows now)
        ann_df = ann_df.reset_index(drop=True)
        dlc_df = dlc_df.reset_index(drop=True)
        merged_df = pd.concat([ann_df, dlc_df], axis=1)

        # Store result
        all_trials_data.append({
            "video_name_with_ext": video_name_with_ext,  # e.g. 'MyVideo.mp4'
            "trial_number": trial_num,
            "annotation_file": fname,
            "dlc_file": dlc_filename,
            "merged_data": merged_df
        })

    return all_trials_data


##############################################################################
# 3. Utilities to Combine Trials and Save
##############################################################################

def combine_all_trials(processed_trials):
    """
    Concatenates all trial DataFrames into a single big DataFrame,
    adding columns for the video name and trial number to keep track.
    """
    df_list = []
    for trial_info in processed_trials:
        df = trial_info["merged_data"].copy()
        df["video_name_with_ext"] = trial_info["video_name_with_ext"]
        df["trial_number"] = trial_info["trial_number"]
        df_list.append(df)

    if not df_list:
        return pd.DataFrame()  # return empty if no valid trials

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df

##############################################################################
# 4. Main: Run Preprocessing, Combine, and Save to CSV
##############################################################################

if __name__ == "__main__":
    # 1) Preprocess to get a list of trials (each with 360 frames)
    processed_trials = preprocess_data()

    print(f"\nFinished preprocessing. Found {len(processed_trials)} valid trial(s).")
    if not processed_trials:
        print("No trials to combine. Exiting.")
        exit(0)

    # 2) Combine all trial data into one DataFrame
    combined_df = combine_all_trials(processed_trials)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    # 3) Save combined data to CSV
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved combined trial data to '{OUTPUT_CSV}'.")

    # Optional: show first few rows
    print("\nPreview of combined DataFrame:\n", combined_df.head())
