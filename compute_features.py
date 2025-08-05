import pandas as pd
import numpy as np
from scipy import signal
import mne
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
from matplotlib import pyplot as plt
from fooof import FOOOF, Bands
import py_neuromodulation as nm

from preprocess_data import preprocess_data
from compute import run_py_neuro, compute_directionality, compute_bursts


f_bands = [[0, 4], [4, 8], [8, 15], [15, 30], [30, 55]]
f_bands_names = ["delta", "theta", "alpha", "beta", "gamma"]
fooof_bands = Bands({"delta": [0, 4], "theta": [4, 8], "alpha": [8, 15], "beta": [15, 30], "gamma": [30, 55]})
sum_range = [1, 80]
fooof_range = [10, 50]
path_out = "features_out"
out_path_py_neuro = "py_neuro_out"
PATH_READ = "df_not_na_leftruns"

df = pd.read_csv("map_scores/scores_date_mapped.csv")
df_art_annotations = pd.read_excel("annotation_artifacts_rcs.xlsx", sheet_name="Sheet1")
files_leave_out = df_art_annotations.query("Index == 'ALL'")["Filename"].unique().tolist()
files_duplicates = list(df_art_annotations["Duplicate"][df_art_annotations["Duplicate"].notna()])
files = [f for f in os.listdir(PATH_READ) if f.endswith(".pkl") and f[:-4] not in files_duplicates and f[:-4] not in files_leave_out]


def compute_file(file_):
    sub = file_.split("_")[0]
    # check if output file already exists
    if os.path.exists(os.path.join(path_out, file_.replace(".pkl", "_features_prep.csv"))):
        print(f"Skipping {file_} because output file already exists.")
        return

    try:
        with open(os.path.join(PATH_READ, file_), "rb") as file:
            d_sub = pickle.load(file)
    except Exception as e:
        print(f"Error reading {file_}: {e}")
        return

    # it's a list of dfs
    keys_ = [f for f in list(d_sub.keys()) if f != "fs" and type(f) == np.int64]
    fs = d_sub["fs"]
    # if the file is less than 5s long, skip
    if len(d_sub[keys_[0]]) < fs * 5:
        print(f"Skipping {file_} because it is less than 5 seconds long.")
        return
    list_scores = [k for k in list(d_sub.keys()) if type(k) != np.int64 and k != "fs"]
    score_values = [d_sub[k] for k in list_scores]
    #score = d_sub["score"]
    d_features = []
    spectra_ = {}
    coh_spectra = {}
    for k in keys_:
        file_short = file_[:-4]
        if int(k) in list(df_art_annotations.query("Filename == @file_short")["Index"]):
            continue
        df_d_idx = d_sub[k]

        data_l = []
        chs = [c for c in df_d_idx.columns if c.startswith("SC_") or c.startswith("C_")]
        for ch in df_d_idx.columns:
            if ch.startswith("Amplitude"):
                continue
            data_ = df_d_idx[ch].values
            data_filtered = preprocess_data(data_, fs)

            data_l.append(data_filtered)
            data_ = data_filtered

            f, Zxx_ = signal.welch(data_, fs=fs, nperseg=fs)
            Zxx = np.log(Zxx_)
            if k not in spectra_:
                spectra_[k] = {}
            spectra_[k][ch] = (f, Zxx)

            # plt.figure()
            # plt.plot(f, Zxx, linewidth=0.5)
            # plt.savefig("test_psd.pdf")
            
            sum_range_idx_ = np.where((f >= sum_range[0]) & (f <= sum_range[1]))[0]
            total_power = np.sum(Zxx[sum_range_idx_])

            fm = FOOOF()
            fm.fit(f, Zxx_, freq_range=fooof_range)
            offset, exponent = fm.aperiodic_params_
            d_features.append({
                "subject": sub,
                "channel": ch,
                "idx": k,
                "feature_name" : "offset",
                "feature_value": offset,
            })
            d_features.append({
                "subject": sub,
                "channel": ch,
                "idx": k,
                "feature_name" : "exponent",
                "feature_value": exponent,
            })

            for i, (f_band, f_band_name) in enumerate(zip(f_bands, f_bands_names)):
                idx_band = np.where((f >= f_band[0]) & (f <= f_band[1]))[0]
                if len(idx_band) == 0:
                    continue
                band_power = np.mean(Zxx[idx_band]) / total_power
                d_features.append({
                    "subject": sub,
                    "channel": ch,
                    "idx": k,
                    "feature_name": f_band_name,
                    "feature_value": band_power,
                })
            
        data_pynm = np.array(data_l)

        ### BURST COMPUTATION
        for ch_idx, ch in enumerate(chs):
            for i, (f_band, f_band_name) in enumerate(zip(f_bands, f_bands_names)):
                try:
                    bursts_df = compute_bursts(data_pynm[ch_idx].copy(), fs, filter_low=f_band[0], filter_high=f_band[1])
                    d_features.append({
                        "subject": sub,
                        "channel": ch,
                        "idx": k,
                        "feature_name": f"burst_duration_{f_band_name}_ms",
                        "feature_value": bursts_df["duration"].mean()*20,
                    })
                    d_features.append({
                        "subject": sub,
                        "channel": ch,
                        "idx": k,
                        "feature_name": f"burst_amplitude_{f_band_name}",
                        "feature_value": bursts_df["mean_amplitude"].mean(),
                    })
                except Exception as e:
                    print(f"Error computing bursts for {file_}, channel {ch}, band {f_band_name}: {e}")
                    continue

         ### Py-NEURO FEATURE COMPUTATION
        features = run_py_neuro(data_pynm.copy(), file_, fs, chs, fooof_range, k, out_path_py_neuro=out_path_py_neuro)

        if features is None:
            continue

        chs_sc_left = [c for c in chs if c.startswith("SC_") and c.endswith("_left")]
        chs_sc_right = [c for c in chs if c.startswith("SC_") and c.endswith("_right")]
        chs_c_left = [c for c in chs if c.startswith("C_") and c.endswith("_left")]
        chs_c_right = [c for c in chs if c.startswith("C_") and c.endswith("_right")]
        
        combs = [
            (chs_sc_left, chs_sc_right),
            (chs_c_left, chs_c_right),
            (chs_sc_left, chs_c_left),
            (chs_sc_right, chs_c_right),
            (chs_sc_left, chs_c_right),
            (chs_sc_right, chs_c_left),
        ]
        comb_names = [
            "SC_left_right",
            "C_left_right",
            "SC_left_C_left",
            "SC_right_C_right",
            "SC_left_C_right",
            "SC_right_C_left",
        ]
        for i, (chs1, chs2) in enumerate(combs):
            corrs_ex, corrs_off, coh_, f_coh_ = compute_directionality(data_pynm, fs, chs, chs1, chs2, features)
            if len(corrs_ex) > 0:
                d_features.append({
                    "subject": sub,
                    "channel": comb_names[i],
                    "idx": k,
                    "feature_name": f"fooof_a_exp_corr_{comb_names[i]}",
                    "feature_value": np.mean(corrs_ex),
                })
                d_features.append({
                    "subject": sub,
                    "channel": comb_names[i],
                    "idx": k,
                    "feature_name": f"fooof_a_offset_corr_{comb_names[i]}",
                    "feature_value": np.mean(corrs_off),
                })

                if k not in coh_spectra:
                    coh_spectra[k] = {}
                coh_spectra[k][comb_names[i]] = (f_coh_, np.array(coh_).mean(axis=0))

        features_median = features.median(axis=0)
        cols_features_py_neuro_add = [c for c in features.columns if c != "time"]
        for col in cols_features_py_neuro_add:
            if "fooof_a_" in col:
                col_name_median = f"{col}_median"
            else:
                col_name_median = col
            if "left" in col:
                ch_name_pyneuro_feature = col[:col.find("left")+4]
            elif "right" in col:
                ch_name_pyneuro_feature = col[:col.find("right")+5]
            d_features.append({
                "subject": sub,
                "channel": ch_name_pyneuro_feature,
                "idx": k,
                "feature_name": col_name_median[len(ch_name_pyneuro_feature)+1:],
                "feature_value": features_median[col].mean(),
            })

    d_features_comb = pd.DataFrame(d_features)
    d_features_comb["fs"] = fs

    for score_col in list_scores:
        d_features_comb[score_col] = score_values[list_scores.index(score_col)]

    d_features_comb.to_csv(os.path.join(path_out, file_.replace(".pkl", "_features_prep.csv")), index=False)

    # use pickle to save the spectra and coherence spectra
    with open(os.path.join(path_out, file_.replace(".pkl", "_spectra.pkl")), "wb") as f:
        pickle.dump(spectra_, f)
    with open(os.path.join(path_out, file_.replace(".pkl", "_coh_spectra.pkl")), "wb") as f:
        pickle.dump(coh_spectra, f)

# use joblib to run in parallel
compute_file(files[0])  # Test with the first file to ensure everything works
Parallel(n_jobs=-1, verbose=0)(delayed(compute_file)(file_) for file_ in tqdm(files, desc="Computing features"))