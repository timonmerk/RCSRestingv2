import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy import signal, stats
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed


def run_day(num_days=0):
    PATH_ = "/Users/Timon/Downloads/df_2025-07-14_oura_both_hems.parq"

    df = pd.read_parquet(PATH_)

    subs = df["pt_id"].unique().tolist()
    features_ = []

    # use B001 just till day 100

    for sub in tqdm(subs):
        #if sub != "B008":
        #    continue
        df_sub = df.query("pt_id == @sub")
        days = df_sub["timestamp"].dt.date.unique().tolist()
        response_ = df_sub["state_label_str"]
        response = [] # Responder vs 
        data_day = []
        if sub == "B001":
            days = days[:149]

        f_ = []
        response_day = []
        for day in days:
            idx = df_sub["timestamp"].dt.date == day
            if idx.sum() > 0:
                response = response_[idx].values[0]
                # leave out Disinhibited and Transition phases
                if response != "Responder" and response != "Pre-DBS" and response != "Non-Responder":
                    continue

            # get one week before
            idx_week = np.logical_and(df_sub["timestamp"].dt.date >= (day - pd.Timedelta(days=num_days)), df_sub["timestamp"].dt.date <= day)

            series_week = df_sub[idx_week][["timestamp", "lfp_right_OvER_interpolate",
                                            "lfp_left_OvER_interpolate",
                                            "delta_lfp_left_OvER_interpolate_dayR2"]].set_index("timestamp")
            series_week_resample = series_week.resample("30min").mean().interpolate(method="linear")

            if series_week_resample.shape[0] != (num_days+1)*24*2:
                continue
            data = series_week_resample.values
            time = series_week_resample.index

            power_6h_left = None
            power_12h_left = None
            power_24h_left = None
            power_6h_left_normed = None
            power_12h_left_normed = None
            power_24h_left_normed = None
            Hjorth_activity_left = None
            Hjorth_mobility_left = None
            Hjorth_complexity_left = None
            power_6h_right = None
            power_12h_right = None
            power_24h_right = None
            power_6h_right_normed = None
            power_12h_right_normed = None
            power_24h_right_normed = None
            Hjorth_activity_right = None
            Hjorth_mobility_right = None
            Hjorth_complexity_right = None
            Pxx_left = None
            Pxx_right = None
            power_left_sum = None
            power_right_sum = None
            delta_lfp_left_OvER_interpolate_dayR2 = None

            for data_idx in range(data.shape[1]):
                if np.nan_to_num(data[:, data_idx]).sum() != 0:
                    data[:, data_idx] = stats.zscore(data[:, data_idx])
                    f, Pxx = welch(data[:, data_idx], fs=1, nperseg=data.shape[0])
                    Pxx = Pxx[1:][::-1]
                    # Pxx = stats.zscore(Pxx[1:][::-1])
                    time_periods = np.array((30/60)*1/f[1:]).round(2)[::-1]
                    if data_idx == 0:
                        Pxx_right = Pxx
                    else:
                        Pxx_left = Pxx

                    idx_6_h = np.where(time_periods == 6)[0]
                    idx_12_h = np.where(time_periods == 12)[0]
                    idx_24_h = np.where(time_periods == 24)[0]

                    

                    if data_idx == 1:  # left hemisphere
                        power_6h_left = Pxx[idx_6_h][0] if idx_6_h.size > 0 else np.nan
                        power_12h_left = Pxx[idx_12_h][0] if idx_12_h.size > 0 else np.nan
                        power_24h_left = Pxx[idx_24_h][0] if idx_24_h.size > 0 else np.nan
                        power_6h_left_normed = power_6h_left / Pxx.sum() if Pxx.sum() != 0 else np.nan
                        power_12h_left_normed = power_12h_left / Pxx.sum() if Pxx.sum() != 0 else np.nan
                        power_24h_left_normed = power_24h_left / Pxx.sum() if Pxx.sum() != 0 else np.nan
                        power_left_sum = Pxx_left.sum()

                        Hjorth_activity_left = np.var(data[:, data_idx])
                        Hjorth_mobility_left = np.sqrt(np.var(np.diff(data[:, data_idx]))) / Hjorth_activity_left
                        Hjorth_complexity_left = np.sqrt(np.var(np.diff(np.diff(data[:, data_idx])))) / np.var(np.diff(data[:, data_idx])) / np.sqrt(Hjorth_activity_left)
                        delta_lfp_left_OvER_interpolate_dayR2 = np.nanmean(data[:, 2])
                    elif data_idx == 0:  # right hemisphere
                        #power_7h_right = Pxx[idx_7_h][0] if idx_7_h.size > 0 else np.nan
                        power_12h_right = Pxx[idx_12_h][0] if idx_12_h.size > 0 else np.nan
                        power_24h_right = Pxx[idx_24_h][0] if idx_24_h.size > 0 else np.nan
                        power_12h_right_normed = power_12h_right / Pxx.sum() if Pxx.sum() != 0 else np.nan
                        power_24h_right_normed = power_24h_right / Pxx.sum() if Pxx.sum() != 0 else np.nan
                        power_right_sum = Pxx_right.sum()

                        Hjorth_activity_right = np.var(data[:, data_idx])
                        Hjorth_mobility_right = np.sqrt(np.var(np.diff(data[:, data_idx]))) / Hjorth_activity_right
                        Hjorth_complexity_right = np.sqrt(np.var(np.diff(np.diff(data[:, data_idx])))) / np.var(np.diff(data[:, data_idx])) / np.sqrt(Hjorth_activity_right)

            features_.append({
                "subject": sub,
                "date": day,
                "response": response,
                "power_6h_left": power_6h_left,
                "power_12h_left": power_12h_left,
                "power_24h_left": power_24h_left,
                "power_6h_left_normed": power_6h_left_normed,
                "power_12h_left_normed": power_12h_left_normed,
                "power_24h_left_normed": power_24h_left_normed,
                "Hjorth_activity_left": Hjorth_activity_left,
                "Hjorth_mobility_left": Hjorth_mobility_left,
                "Hjorth_complexity_left": Hjorth_complexity_left,
                "power_6h_right": power_6h_right,
                "power_12h_right": power_12h_right,
                "power_24h_right": power_24h_right,
                "power_6h_right_normed": power_6h_right_normed,
                "power_12h_right_normed": power_12h_right_normed,
                "power_24h_right_normed": power_24h_right_normed,
                "Hjorth_activity_right": Hjorth_activity_right,
                "Hjorth_mobility_right": Hjorth_mobility_right,
                "Hjorth_complexity_right": Hjorth_complexity_right,
                "power_left_sum": power_left_sum,
                "power_right_sum": power_right_sum,
                "delta_lfp_left_OvER_interpolate_dayR2" : delta_lfp_left_OvER_interpolate_dayR2,
                "Pxx_left" : Pxx_left,
                "Pxx_right": Pxx_right
            })

        # plt.figure()
        # plt.plot(time_periods, Pxx_left[::-1])
        # plt.xscale("log")
        # plt.show(block=True)

        # if Pxx_left or Pxx_right:
        #     Pxx_sub_left = np.array(Pxx_left)
        #     Pxx_sub_right = np.array(Pxx_right)
        #     Pxx_subs_left.append(Pxx_sub_left)
        #     Pxx_subs_right.append(Pxx_sub_right)

    df_features = pd.DataFrame(features_)
    df_features.to_csv(f"else/df_features_rick_{num_days+1}days_normed_by_sum.csv")

    with open(f"else/time_periods_{num_days+1}days.pkl", "wb") as f:
        pickle.dump(time_periods, f)

if __name__ == "__main__":
    days = np.arange(0, 15)
    #run_day(0)  # run for 1 day first
    # use joblib to parallelize the computation
    Parallel(n_jobs=-1)(delayed(run_day)(num_days=day) for day  in days)

# zs_ = []
# plt.figure()
# for Px_ in Pxx_sub_left:
#     plt.plot(time_periods, stats.zscore(Px_), alpha=0.1, color="gray")
#     zs_.append(stats.zscore(Px_))
# plt.plot(time_periods, stats.zscore(np.median(Pxx_sub_left, axis=0)))
# plt.xscale("log")
# plt.xlim(1, 40)
# plt.xlabel("Time period [h]")
# plt.ylabel("Power [a.u.]")
# #plt.yscale("log")
# plt.show(block=True)


# plt.figure()
# plt.imshow(
#     np.array(zs_),
#     aspect="auto",
#     cmap="viridis",
#     origin="lower"
# )
# #plt.xscale("log")
# plt.clim(-2, 2)


# plt.show(block=True)
# plt.colorbar(label="Z-Score")
# plt.xlabel("Time Periods (hours)")
# plt.ylabel("Days")
# plt.xticks(rotation=45)

# plt.title(f"Subject {sub} - Response: {response[0]}")
# plt.tight_layout()


    # # Define scales (adjust range depending on your data resolution and interest)
    # scales = np.arange(1, 128)

    # # Use real Morlet wavelet ('morl') — great for time-frequency analysis
    # coefficients, frequencies = pywt.cwt(data, scales, 'morl')

    # frequencies_per_hour = frequencies * (24)

    # sampling_period = 1/30*60
    # f = scale2frequency("morl", scales)/sampling_period
    # # Plot scalogram
    # plt.figure(figsize=(12, 6))
    # plt.imshow(
    #     np.abs(coefficients),
    #     extent=[time[0], time[-1], frequencies_per_hour[-1], frequencies_per_hour[0]],
    #     cmap='viridis',
    #     aspect='auto'
    # )
    # plt.colorbar(label='|CWT Coefficient|')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time')
    # plt.title('CWT using Morlet Wavelet')
    # plt.xticks(rotation=45)
    # # flip y-axis
    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    # plt.show()

    # coeff_ = np.abs(coefficients)[:, :, 0].mean(axis=1)
    # plt.figure()
    # plt.plot((1/f)*30/60, coeff_)
    # plt.show(block=True)

    