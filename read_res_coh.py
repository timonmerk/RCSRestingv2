import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats

df_annot = pd.read_excel("annotation_artifacts_rcs.xlsx", sheet_name="Sheet1")
duplicates = df_annot["Duplicate"].dropna().unique()
all_wrong = df_annot.query("Index == 'ALL'")["Filename"].unique()

f_bands = [[0, 4], [4, 8], [8, 15], [15, 30], [30, 55]]
f_bands_names = ["delta", "theta", "alpha", "beta", "gamma"]

df_scores = pd.read_csv("features_prep_combined.csv")
df_scores = df_scores.groupby(["subject", "file"])["score"].mean().reset_index()

PLT_ = False
PATH_READ = "features_out_with_bursts"

files = [f for f in os.listdir(PATH_READ) if "coh_spectra.pkl" in f]

df_ = pd.DataFrame()

for file in tqdm(files):
    f_name_find = file[:file.find("_coh_spectra")]
    if f_name_find in duplicates or f_name_find in all_wrong:
        continue
    with open(os.path.join("features_out", file), "rb") as f:
        coh_spectra = pickle.load(f)
    sub = file.split("_")[0]

    combinations = {
        "SC_left_right": [], "C_left_right": [], "SC_left_C_left": [], "SC_right_C_right": [],
        "SC_left_C_right": [], "SC_right_C_left": [],
    }
    df_annot_file = df_annot.query("Filename == @f_name_find")
    for key in coh_spectra.keys():
        if key in df_annot_file["Index"].values:
            continue
        for comb in combinations.keys():
            if comb in coh_spectra[key].keys():
                f, coh = coh_spectra[key][comb]
                combinations[comb].append(coh)

    valid_combs = [k for k in combinations if len(combinations[k]) > 0]
    n_combs = len(valid_combs)

    res_return = {"SC_left_right": None, "C_left_right": None,
                "SC_left_C_left": None, "SC_right_C_right": None,
                "SC_left_C_right": None, "SC_right_C_left": None}
    for comb in valid_combs:
        arr = np.array(combinations[comb])
        # median_coh = np.median(arr, axis=0)[:100]
        median_coh = arr[arr.shape[0] // 2, :100]
        cols_median_coh = np.arange(100)
        df_add = pd.DataFrame(np.expand_dims(median_coh, 0), columns=cols_median_coh)
        df_add["subject"] = sub
        df_add["key"] = key
        df_add["file"] = file
        df_add["combination"] = comb
        df_add["score"] = df_scores.query("subject == @sub and file == @f_name_find")["score"].values[0]

        for band, band_name in zip(f_bands, f_bands_names):
            band_coh = np.mean(median_coh[band[0]:band[1]])
            df_add[f"{band_name}"] = band_coh

        df_ = pd.concat([df_, df_add], ignore_index=True)

    if PLT_:
        fig, axs = plt.subplots(n_combs, 1, figsize=(12, 4 * n_combs), sharex=True)
        if n_combs == 1:
            axs = [axs]  # Ensure it's always iterable
        for idx, comb in enumerate(valid_combs):
            arr = np.array(combinations[comb])
            for i in range(arr.shape[0]):
                axs[idx].plot(f, arr[i, :], linewidth=0.5)
            axs[idx].set_title(f"{comb} Coherence")
            axs[idx].set_ylabel("Coherence")
            axs[idx].set_xlim(0, 100)

        axs[-1].set_xlabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig("all_combinations_coherence.pdf")
        plt.show()

df_.to_csv("coherence_spectra.csv", index=False)

df_corr_out = []
for band in f_bands_names:
    for comb in df_["combination"].unique():
        df_comb = df_.query("combination == @comb")
        if df_comb.empty:
            continue
        for sub in df_["subject"].unique():
            df_sub = df_comb[df_comb["subject"] == sub]
            df_sub = df_sub.groupby("file")[["score", band]].mean().reset_index()
            if df_sub.empty:
                continue
            corr, p_value = stats.pearsonr(df_sub[band], df_sub["score"])
            df_corr_out.append({
                "subject": sub,
                "combination": comb,
                "band": band,
                "correlation": corr,
                "p_value": p_value
            })
        df_all = df_comb.groupby(["file", "subject"])[["score", band]].mean().reset_index()
        corr, p_value = stats.pearsonr(df_all[band], df_all["score"])
        df_corr_out.append({
            "subject": "ALL",
            "combination": comb,
            "band": band,
            "correlation": corr,
            "p_value": p_value
        })
df_corr_out = pd.DataFrame(df_corr_out)
df_corr_out.to_csv("coherence_correlation.csv", index=False)

plt.figure(figsize=(8, 6))
for comb_idx, comb in enumerate(df_corr_out["combination"].unique()):
    df_comb = df_corr_out.query("combination == @comb")
    #for band_idx, band in enumerate(f_bands_names):
    plt.subplot(2, 3, comb_idx+1)
    sns.boxplot(data=df_comb.query("subject != 'ALL'"), x="band", y="correlation", showmeans=True, showfliers=False, boxprops=dict(facecolor='none', edgecolor='black'))
    sns.swarmplot(data=df_comb.query("subject != 'ALL'"), x="band", y="correlation", color=".25", alpha=0.5)
    sns.swarmplot(data=df_comb.query("subject == 'ALL'"), x="band", y="correlation", color="red", alpha=1)
    plt.title(f"{comb}")
    plt.xticks(rotation=90)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/coherence_correlation_by_combination.pdf")

plt_cnt = 0
plt.figure(figsize=(20, 20))  # 8 x 4 plot
for sub in sorted(df_["subject"].unique()):
    for idx, combination in enumerate(df_["combination"].unique()):
        plt_cnt += 1
        plt.subplot(8, 6, plt_cnt)
        df_sub = df_[(df_["subject"] == sub) & (df_["combination"] == combination)]
        plt.title(f"{combination}")
        if df_sub.empty:
            continue
        #for row in range(df_sub.shape[0]):
        #    plt.plot(df_sub.iloc[row, :-6].values, color="black", alpha=0.5)
        # plot mean and std
        mean_coh = df_sub.iloc[:, :-12].mean().values
        std_coh = df_sub.iloc[:, :-12].std().values
        plt.plot(mean_coh, color="black", linewidth=2, )
        plt.fill_between(np.arange(mean_coh.shape[0]), mean_coh - std_coh, mean_coh + std_coh, color="black", alpha=0.2,)
        #plt.legend()
        plt.xlabel("Frequency (Hz)")
        if idx == 0:
            plt.ylabel(sub)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/coherence_all_subjects.pdf")
plt.close()

plt.figure(figsize=(8, 6))
for comb_idx, comb in enumerate(df_corr_out["combination"].unique()):
    plt.subplot(2, 3, comb_idx+1)
    df_comb = df_[df_["combination"] == comb]
    plt.title(f"{comb}")
    if df_comb.empty:
        continue
    #for row in range(df_sub.shape[0]):
    #    plt.plot(df_sub.iloc[row, :-6].values, color="black", alpha=0.5)
    # plot mean and std
    mean_coh = []
    for sub in sorted(df_comb["subject"].unique()):
        df_sub = df_comb[df_comb["subject"] == sub]
        if df_sub.empty:
            continue
        mean_coh_sub = df_sub.iloc[:, :-12].mean().values
        mean_coh.append(mean_coh_sub)
        plt.plot(mean_coh_sub, color="black", linewidth=0.5, alpha=0.5)
    mean_coh_all = np.mean(mean_coh, axis=0)
    std_coh_all = np.std(mean_coh, axis=0)
    plt.plot(mean_coh_all, color="black", linewidth=2, )
    plt.fill_between(np.arange(mean_coh_all.shape[0]), mean_coh_all - std_coh_all, mean_coh_all + std_coh_all, color="black", alpha=0.2,)
    plt.xlabel("Frequency (Hz)")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/coherence_all_subjects_combined.pdf")
plt.close()