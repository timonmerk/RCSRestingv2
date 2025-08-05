import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
import mne
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle


def plot_df_timeseries(df_plt: pd.DataFrame, out_path: str, FILTER = False, resample:bool=False, PLOT_=False):

    if PLOT_:
        pdf_ = PdfPages(out_path)
    TIME_INTERVAL = 10
    
    fs = int(1 / np.unique(df_plt["localTimeUnix"].diff(), return_counts=True)[0][0]*1000)
    samples = fs * TIME_INTERVAL

    if resample:
        if fs != 250:
            # resample to 250 Hz
            df_plt["localTimeUnix"] = pd.to_datetime(df_plt["localTimeUnix"], unit="ms")
            df_plt = df_plt.set_index("localTimeUnix").resample("4ms").mean().reset_index()
            fs = 250
    
    df_plt["idx_counter"] = np.arange(df_plt.shape[0]) // samples
    chs = [c for c in df_plt.columns if c.startswith("SC_") or c.startswith("C_")]
    cols_stim = [c for c in df_plt.columns if c.startswith("Amplitude_mA")]
    d_sub = {}

    for idx_cnt in df_plt["idx_counter"].unique():
        df_range_ = df_plt[df_plt["idx_counter"] == idx_cnt]
        # if there is any NaN in any of the channels, skip this idx
        if df_range_[chs].isna().any().any():
            #print(f"Skipping idx {idx_cnt} due to NaN values in channels {chs}")
            continue
        d_sub[idx_cnt] = df_range_[chs + cols_stim].copy()

        if PLOT_ is False:
            continue
        if len(chs) == 1 or len(chs) == 2 or len(chs) == 3:
            plt.figure(figsize=(12, 5))
        if len(chs) == 4 or len(chs) == 5:
            plt.figure(figsize=(12, 10))
        if len(chs) == 6 or len(chs) == 7 or len(chs) == 8:
            plt.figure(figsize=(12, 15))
        
        plt.suptitle(f"Time Series - idx: {idx_cnt}\n" +
                        f"{chs}\n"+
                        f"{df_range_.index[0]} - {df_range_.index[-1]}")
        for ch_idx, ch_name in enumerate(chs):
            data_ = df_range_[ch_name].to_numpy()
            # fill nans with 0
            if FILTER:
                data_raw = np.nan_to_num(data_, nan=0.0)
                # if fs > 250:
                #     f_low = 140
                #     f_high = 160
                # else:
                f_low = 105
                f_high = 95
                data_filtered = mne.filter.filter_data(
                    data_raw,
                    sfreq=fs,
                    l_freq=f_low,
                    h_freq=f_high,
                    method='iir',
                    verbose=False
                )

                # lop pass filter until 100 HZ
                data_filtered = mne.filter.filter_data(
                    data_filtered,
                    sfreq=fs,
                    l_freq=None,
                    h_freq=100,
                    method='iir',
                    verbose=False
                )
                data_filtered = mne.filter.filter_data(
                    data_filtered,
                    sfreq=fs,
                    l_freq=65,
                    h_freq=55,
                    method='iir',
                    verbose=False
                )
                data_filtered = mne.filter.filter_data(
                    data_filtered,
                    sfreq=fs,
                    l_freq=0.5,
                    h_freq=None,
                    method='iir',
                    verbose=False
                )
                data_ = data_filtered
            
            plt.subplot(len(chs), 3, ch_idx*3 + 1)
            plt.plot(np.arange(0, data_.shape[0]/fs, 1/fs), data_, linewidth=0.5)
            plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
            plt.xlabel("Time [s]")
            plt.ylabel(ch_name)
            plt.subplot(len(chs), 3, ch_idx*3 + 2)
            plt.plot(np.arange(0, data_[:fs].shape[0]/fs, 1/fs), data_[:fs], linewidth=0.5)
            plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude [a.u.]")
            plt.subplot(len(chs), 3, ch_idx*3 + 3)
            if FILTER is False:
                data_ = np.nan_to_num(data_, nan=0.0)
            f, Pxx = signal.welch(data_, fs=fs, nperseg=fs)
            plt.plot(f, np.log(Pxx), linewidth=0.5)
            plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD [a.u.]")
        plt.tight_layout()
        pdf_.savefig(bbox_inches='tight')
        plt.close()
    if PLOT_:
        pdf_.close()

    # if d_sub is not empty, save it as a pickle file
    if d_sub:
        d_sub["fs"] = fs
        for score_col in scores.columns[:-2]:
            d_sub[score_col] = scores[scores["file"] == os.path.basename(out_path.replace("_timeseries.pdf", ".csv"))][score_col].values[0]
        #d_sub["score"] = scores[scores["file"] == os.path.basename(out_path.replace("_timeseries.pdf", ".csv"))]["ybocs"].values[0]
        with open(os.path.join(PATH_DATA_OUT, os.path.basename(out_path.replace(".pdf", ".pkl"))), "wb") as f:
            pickle.dump(d_sub, f)
    else:
        # delete the pdf
        if os.path.exists(out_path):
            os.remove(out_path)
            print(f"Deleted empty file: {out_path}")
        else:
            print(f"File {out_path} does not exist, nothing to delete.")


PATH_BASE = "/scratch/timonmerk"
PATH_ = os.path.join(PATH_BASE, "rs_prep")  # liegt aktuell in elias timonmerk/rs_prep
PATH_OUT_PRE = os.path.join(PATH_BASE, "OCDRCSResting/rs_viz_no_nan")  # rs_viz_ts_non_filtered
PATH_OUT = os.path.join(PATH_BASE, "OCDRCSResting/rs_viz_no_nan_leftruns")
PATH_DATA_OUT = "df_not_na_leftruns"
PATH_DATA_PRE  = "df_not_na"

files = [f for f in os.listdir(PATH_) if f.endswith(".csv")]
scores = pd.read_csv("map_scores/files_ybocs.csv")
files = [f for f in files if f in scores["file"].values]

out_left = []

for f in files:
    out_path_check = f.replace(".csv", "_timeseries.pkl")
    out_path_check = os.path.join(PATH_DATA_PRE, out_path_check)
    if os.path.exists(out_path_check) is False:
        out_left.append(f)
        



def process_file(f):
    try:
        df = pd.read_csv(os.path.join(PATH_, f))
    except Exception as e:
        print(f"Error reading {f}: {e}")
        return
    if df.empty:
        print(f"File {f} is empty. Skipping.")
        return
    out_ = os.path.join(PATH_OUT, f.replace(".csv", "_timeseries.pdf"))
    try:
        plot_df_timeseries(df, out_path=out_, FILTER=False, resample=False, PLOT_=True)
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return

# for f in tqdm(out_left):
#     process_file(f)


Parallel(n_jobs=10)(
   delayed(process_file)(f) for f in tqdm(out_left, desc="Processing files")
)

# Run in parallel using all available cores
# shuffle files
# import random
# random.shuffle(files)
# for f in files:
#     print(f"Processing file: {f}")
#     process_file(f)   # Test with the first file to ensure the function works


# for f in tqdm(files, desc="Processing files"):
#     df = pd.read_csv(os.path.join(PATH_, f))
#     out_ = os.path.join(PATH_OUT, f.replace(".csv", "_timeseries.pdf"))
#     plot_df_timeseries(df, out_path=out_, FILTER=True)