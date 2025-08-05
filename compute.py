import py_neuromodulation as nm
import os
import numpy as np
from scipy import signal
import pandas as pd
import mne
from matplotlib import pyplot as plt
from scipy.signal import hilbert



def run_py_neuro(data_pynm, file_, fs, chs: list, fooof_range: list, k, out_path_py_neuro: str ="py_neuro_out"):

    settings = nm.NMSettings.get_fast_compute()
    settings = settings.reset()

    settings.features.sharpwave_analysis = True
    settings.features.raw_hjorth = True
    settings.features.fooof = True
    settings.features.fft = True
    settings.fft_settings.return_spectrum = True
    settings.fooof_settings.freq_range_hz = fooof_range
    settings.sharpwave_analysis_settings.filter_ranges_hz = [[1, 5], [1, 12]]
    settings.features.return_raw = True
    settings.sampling_rate_features_hz = 1
    settings.preprocessing = []
    settings.postprocessing.feature_normalization = False
    settings.fooof_settings.windowlength_ms = 1000

    settings.frequency_ranges_hz["low_gamma"] = (40, 55)
    settings.frequency_ranges_hz["high_gamma"] = (65, 95)

    channels = nm.utils.set_channels(
        ch_names=chs,
        ch_types=["dbs" for _ in chs],  # assuming all channels are DBS
        reference=None,
        new_names="default",
        used_types=("ecog", "dbs", "seeg", "eeg"),
        target_keywords=None,
    )

    stream = nm.Stream(
        sfreq=fs,
        channels=channels,
        settings=settings,
        line_noise=60,
        coord_list=None,
        coord_names=None,
        verbose=False,
    )

    try:
        features = stream.run(
            data=data_pynm,
            out_dir=out_path_py_neuro,
            experiment_name=f"{os.path.basename(file_)[:-4]}_{k}",
            save_csv=True,
        )
    except Exception as e:
        print(f"Error processing {file_} for idx {k}: {e}")
        features = None
    return features


def compute_directionality(data_pynm, fs, chs, chs1, chs2, features):
    corrs_ex = []
    corrs_off = []
    coh_ = []
    f_coh_ = None
    if len(chs1) > 0 and len(chs2) > 0:
        for ch1 in chs1:
            dat_ch1 = data_pynm[chs.index(ch1), :]
            
            for ch2 in chs2:
                corr_ap_ex = np.corrcoef(np.nan_to_num(features[f"{ch1}_fooof_a_exp"].values),
                            np.nan_to_num(features[f"{ch2}_fooof_a_exp"].values))
                corr_ap_off = np.corrcoef(np.nan_to_num(features[f"{ch1}_fooof_a_offset"].values),
                            np.nan_to_num(features[f"{ch2}_fooof_a_offset"].values))
                corrs_ex.append(corr_ap_ex[0, 1])
                corrs_off.append(corr_ap_off[0, 1])

                dat_ch2 = data_pynm[chs.index(ch2), :]

                f_coh, coh = signal.coherence(dat_ch1, dat_ch2, fs=fs, nperseg=fs)
                f_coh_ = f_coh
                # plt.figure()
                # plt.plot(f_coh, coh, linewidth=0.5)
                # plt.savefig("test_psd.pdf")
                coh_.append(coh)

    return corrs_ex, corrs_off, np.array(coh_), f_coh_

def get_burst_df(theta_vals_above_thr, theta_vals):
    bursts = []
    in_burst = False
    start_idx = 0

    for idx, val in enumerate(theta_vals_above_thr):
        if val and not in_burst:
            # Start of a new burst
            in_burst = True
            start_idx = idx
        elif not val and in_burst:
            # End of burst
            end_idx = idx
            burst_vals = theta_vals[start_idx:end_idx]
            bursts.append({
                "start": start_idx,
                "end": end_idx - 1,
                "duration": end_idx - start_idx,
                "mean_amplitude": burst_vals.mean()
            })
            in_burst = False

    # Handle case where burst continues to the end
    if in_burst:
        burst_vals = theta_vals[start_idx:]
        bursts.append({
            "start": start_idx,
            "end": len(theta_vals) - 1,
            "duration": len(theta_vals) - start_idx,
            "mean_amplitude": burst_vals.mean()
        })


    # Convert to DataFrame for easy viewing
    bursts_df = pd.DataFrame(bursts)
    #bursts_df["duration"] = bursts_df["duration"] * 0.05
    return bursts_df

def compute_bursts(data, fs, filter_low = 4, filter_high = 8):
    
    raw = mne.io.RawArray(data[np.newaxis, :], mne.create_info(ch_names=["burst_channel"], sfreq=fs, ch_types=["ecog"]))
    raw.filter(l_freq=filter_low, h_freq=filter_high, method='iir', verbose=False)
    raw.resample(sfreq=50, npad="auto", verbose=False)
    data_ = raw.get_data()[0]
    data_f = abs(hilbert(data_))[2:-2]
    #data_f = mne.filter.filter_data(data, sfreq=fs, l_freq=filter_low, h_freq=filter_high, method='iir', verbose=False)
    #data_f = abs(hilbert(mne.filter.filter_data(data, sfreq=fs,
    #    l_freq=10, h_freq=30, method='iir',
    #    verbose=False)))
    # resample to 10 hz


    # raw = mne.io.RawArray(data_f[np.newaxis, :], mne.create_info(ch_names=["burst_channel"], sfreq=fs, ch_types=["ecog"]))
    # fig = raw.plot_psd()
    # plt.savefig("burst_psd.pdf")

    # plt.figure()
    # plt.plot(data)
    # plt.savefig("burst_signal_hilbert.pdf")

    percentile = np.percentile(data_f, 75)
    data_f_above_thr = data_f > percentile

    burst_df = get_burst_df(data_f_above_thr, data_f)

    return burst_df
