import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from datetime import datetime
from tqdm import tqdm


def get_df(f_name):
    """
    Function to read the CSV files and return the dataframes.
    """
    if "left.csv" in f_name:
        f_path_stim = f_name.replace("left.csv", "stim_left.csv")
    elif "right.csv" in f_name:
        f_path_stim = f_name.replace("right.csv", "stim_right.csv")
    elif "NA.csv" in f_name:
        f_path_stim = f_name.replace("NA.csv", "stim_NA.csv")
    df_stim = pd.read_csv(f_path_stim)
    df_stim["localTimeUnix"] = pd.to_datetime(df_stim["Time"]).astype(np.int64)

    df = pd.read_csv(f_name)
    df['localTime'] = pd.to_datetime(df['localTime'])
    df["localTimeUnix"] = df["DerivedTime"]
    df = pd.merge(df, df_stim[["localTimeUnix", "Amplitude_mA"]], on="localTimeUnix", how="left")
    fs = int((1 / np.unique(df["localTimeUnix"].diff(), return_counts=True)[0][0])*1000)
    return df, fs

def find_longest_constant_segment(df, column):
    """
    Find the longest continuous segment in a DataFrame where the values
    in the given column are constant and not NaN.

    Returns:
        segment_info (dict): Information about the segment (value, length, start_idx, end_idx)
        segment_df (DataFrame): The slice of the original DataFrame for that segment
    """
    # Create group IDs for consecutive values
    is_change = (df[column] != df[column].shift()) | df[column].isna()
    df = df.copy()
    df["segment_id"] = is_change.cumsum()

    # Filter out NaNs
    valid_df = df[df[column].notna()]

    # Group and find longest
    segments = (
        valid_df
        .groupby("segment_id")
        .agg(
            value=(column, 'first'),
            length=(column, 'size'),
            start_idx=(column, lambda x: x.index[0]),
            end_idx=(column, lambda x: x.index[-1])
        )
    )

    if segments.empty:
        return None, None

    longest = segments.loc[segments['length'].idxmax()]
    segment_df = df.loc[longest.start_idx : longest.end_idx]

    segment_info = {
        'value': longest.value,
        'length': longest.length,
        'start_idx': longest.start_idx,
        'end_idx': longest.end_idx
    }

    return segment_info, segment_df

PATH_DATA = "/scratch/timonmerk/restingstate_data"
PATH_OUT = "/scratch/timonmerk/rs_prep"
list_files_csv = [os.path.join(PATH_DATA, f) for f in os.listdir(PATH_DATA) if f.endswith('.csv')]

# delete every file that contains "combined" in the name
list_files_combined = [f for f in list_files_csv if "combined" in f]
for f in list_files_combined:
    print(f"Deleting {f}")
    os.remove(f)


f_unique = {}
for f in list_files_csv:
    if "stim" in f:
        continue
    if "left.csv" in f:
        if f[:-len("left.csv")] not in f_unique:
            f_unique[f[:-len("left.csv")]] = ["left"]
        else:
            f_unique[f[:-len("left.csv")]].append("left")
    elif "right.csv" in f:
        if f[:-len("right.csv")] not in f_unique:
            f_unique[f[:-len("right.csv")]] = ["right"]
        else:
            f_unique[f[:-len("right.csv")]].append("right")
    elif "NA.csv" in f:
        if f[:-len("NA.csv")] not in f_unique:
            f_unique[f[:-len("NA.csv")]] = ["NA"]
        else:
            f_unique[f[:-len("NA.csv")]].append("NA")


files_exception = []
for f in tqdm(f_unique.keys()):
    #f = '/scratch/timonmerk/restingstate_data/aDBS005_resting-state1_20201105132400969430_synced_ephys_behav_lfp_v3_'
    df_left = None
    df_right = None
    df_na = None

    if "._" in f:
        print(f"Skipping {f} due to ._ prefix.")
        continue
    try:
        if "left" in f_unique[f]:
            if os.path.exists(os.path.join(PATH_OUT, os.path.basename(f) + "combined.csv")):
                print(f"File {os.path.basename(f)} already exists. Skipping.")
                continue
            df_left, fs_l = get_df(f + "left.csv")
        if "right" in f_unique[f]:
            if os.path.exists(os.path.join(PATH_OUT, os.path.basename(f) + "combined.csv")):
                print(f"File {os.path.basename(f)} already exists. Skipping.")
                continue
            df_right, fs_r = get_df(f + "right.csv")
        if "NA" in f_unique[f]:
            output_file = os.path.basename(f) + "combined_lr.csv"
            if os.path.exists(os.path.join(PATH_OUT, output_file)):
                print(f"File {output_file} already exists. Skipping.")
                continue
            df_na, fs_na = get_df(f + "NA.csv")
    except Exception as e:
        files_exception.append(f)
        print(f"Error reading {f}: {e}")
        continue
    
    if df_na is not None:
        first_valid_idx_na = df_na["Amplitude_mA"].first_valid_index()
        if first_valid_idx_na is not None:
            df_na = df_na.iloc[first_valid_idx_na:].reset_index(drop=True)
            df_na["Amplitude_mA"] = df_na["Amplitude_mA"].ffill()
            _, df = find_longest_constant_segment(df_na, "Amplitude_mA")
            df = df.iloc[10*fs_na:-10*fs_na].reset_index(drop=True)
            cols = [c for c in df.columns if c.startswith("TD_key") or c
                .startswith("Amplitude_mA") or c.startswith("localTimeUnix")]
            df = df[cols]
            df = df.dropna(axis=1, how='all')
            cnt_sc = 0
            for i in range(4):
                if i < 2:
                    hem = "left"
                else:
                    hem = "right"
                if i == 2:
                    cnt_c = 0
                if f"TD_key{i}" in df.columns:
                    df.rename(columns={f"TD_key{i}": f"SC_{cnt_sc}_{hem}"}, inplace=True)
                    cnt_sc += 1
            output_file = os.path.basename(f) + "combined_lr.csv"
            print(f"Writing {output_file}")
            df.to_csv(os.path.join(PATH_OUT, output_file), index=False)
    
    if df_left is not None and df_right is not None:
        # merge the two dataframes on localTimeUnix
        df = pd.merge(df_left, df_right, on="localTimeUnix", suffixes=('_left', '_right'))

        # if df_left["Amplitude_mA"] has a non NaN value before the first df["localTime"], set the first value of df["Amplitude_mA_left"] to that value
        first_time = df["localTimeUnix"].iloc[0]

        # --- For df_left ---
        prior_left = df_left[(df_left["localTimeUnix"] < first_time) & (df_left["Amplitude_mA"].notna())]
        if not prior_left.empty:
            last_left_value = prior_left["Amplitude_mA"].iloc[-1]
            df.at[0, "Amplitude_mA_left"] = last_left_value

        # --- For df_right ---
        prior_right = df_right[(df_right["localTimeUnix"] < first_time) & (df_right["Amplitude_mA"].notna())]
        if not prior_right.empty:
            last_right_value = prior_right["Amplitude_mA"].iloc[-1]
            df.at[0, "Amplitude_mA_right"] = last_right_value
        
        if df["Amplitude_mA_left"].isna().all() and df["Amplitude_mA_right"].isna().all():
            continue

        first_valid_idx_left = df["Amplitude_mA_left"].first_valid_index()
        first_valid_idx_right = df["Amplitude_mA_right"].first_valid_index()
        first_valid_idx = min(i for i in [first_valid_idx_left, first_valid_idx_right] if i is not None)

        df = df.iloc[first_valid_idx:].reset_index(drop=True)

        df["Amplitude_mA_left"] = df["Amplitude_mA_left"].ffill()
        df["Amplitude_mA_right"] = df["Amplitude_mA_right"].ffill()

        if df["Amplitude_mA_left"].isna().all() and df["Amplitude_mA_right"].isna().all() == False:
            df["Amplitude_mA_sum"] = df["Amplitude_mA_right"]
        elif df["Amplitude_mA_right"].isna().all() and df["Amplitude_mA_left"].isna().all() == False:
            df["Amplitude_mA_sum"] = df["Amplitude_mA_left"]
        elif df["Amplitude_mA_left"].isna().all() == False and df["Amplitude_mA_right"].isna().all() == False:
            # both are not NaN, so we can sum them
            df["Amplitude_mA_sum"] = df["Amplitude_mA_left"] + df["Amplitude_mA_right"]
        else:
            print(f"No stim values available for {f}. Skipping.")
            continue

        _, df = find_longest_constant_segment(df, "Amplitude_mA_sum")
        # skip left and right each 10 s
        df = df.iloc[10*fs_l:-10*fs_l].reset_index(drop=True)
        cols = [c for c in df.columns if c.startswith("TD_key") or c.startswith("Amplitude_mA") or c.startswith("localTimeUnix")]
        df = df[cols]
        # contacts 8-9 are generally in medial OFC and 10-11 are in lateral OFC
        # drop columns with only NaN values
        df = df.dropna(axis=1, how='all')
        # replace TD_key0 with SC0 if exists, if not then TD_key1, else exception
        for hem in ["left", "right"]:
            cnt_sc = 0
            for i in range(2):
                if f"TD_key{i}_{hem}" in df.columns:
                    df.rename(columns={f"TD_key{i}_{hem}": f"SC_{cnt_sc}_{hem}"}, inplace=True)
                    cnt_sc += 1
            cnt_c = 0
            for i in range(2, 4):
                if f"TD_key{i}_{hem}" in df.columns:
                    df.rename(columns={f"TD_key{i}_{hem}": f"C_{cnt_c}_{hem}"}, inplace=True)
                    cnt_c += 1

        output_file = os.path.basename(f) + "combined.csv"
        print(f"Writing {output_file}")
        df.to_csv(os.path.join(PATH_OUT, output_file))
