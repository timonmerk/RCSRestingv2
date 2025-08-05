import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.cm as cm


df_annot = pd.read_excel("annotation_artifacts_rcs.xlsx", sheet_name="Sheet1")
duplicates = df_annot["Duplicate"].dropna().unique()
all_wrong = df_annot.query("Index == 'ALL'")["Filename"].unique()
PLT_ = False  # set to False to skip plotting

files = [f for f in os.listdir("features_out") if "_spectra.pkl" in f and "coh" not in f]

df_ = pd.DataFrame()

for file in tqdm(files, desc="Processing spectra"):
    f_name_find = file[:file.find("_spectra")]
    if f_name_find in duplicates or f_name_find in all_wrong:
        continue
    with open(os.path.join("features_out", file), "rb") as f:
        spectra = pickle.load(f)

    sub = file.split("_")[0]
    if sub == "aDBS012":
        print("break")

    ch_combinations = {
        "SC_0_left": [], "SC_1_left": [], "SC_0_right": [], "SC_1_right": [],
        "C_0_left": [], "C_1_left": [], "C_0_right": [], "C_1_right": [],
    }
    df_annot_file = df_annot.query("Filename == @f_name_find")
    for key in spectra.keys():
        if key in df_annot_file["Index"].values:
            continue
        for comb in ch_combinations.keys():
            if comb in spectra[key].keys():
                f_vals_, psd = spectra[key][comb]
                if f_vals_.shape[0] < 250:
                    continue
                if f_vals_.shape[0] == 251 or f_vals_.shape[0] == 501 or f_vals_.shape[0] == 1001:
                    f_vals = f_vals_
                    f_idx_smaller_100 = np.where(f_vals < 100)[0]
                    ch_combinations[comb].append(psd)
                else:
                    # print shape
                    print(f"Unexpected frequency values shape {f_vals_.shape}")
                    continue
    
    valid_combs = [k for k in ch_combinations if len(ch_combinations[k]) > 0]
    n_combs = len(valid_combs)

    for comb in valid_combs:
        arr = np.array(ch_combinations[comb])
        median_psd = arr[arr.shape[0]//2, f_idx_smaller_100]
        df_add = pd.DataFrame(np.expand_dims(median_psd, 0), columns=f_vals[f_idx_smaller_100])
        df_add["subject"] = sub
        df_add["file"] = file
        df_add["combination"] = comb
        df_ = pd.concat([df_, df_add], ignore_index=True)

    if PLT_:
        fig, axs = plt.subplots(n_combs, 1, figsize=(12, 4 * n_combs), sharex=True)
        if n_combs == 1:
            axs = [axs]  # ensure iterable

        for idx, comb in enumerate(valid_combs):
            arr = np.array(ch_combinations[comb])
            for i in range(arr.shape[0]):
                axs[idx].plot(f_vals[f_idx_smaller_100], arr[i][f_idx_smaller_100], linewidth=0.5)
            axs[idx].set_title(f"{comb}")
            axs[idx].set_ylabel("PSD")

        axs[-1].set_xlabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(f"{sub}_all_chs_spectra.pdf")
        plt.close()

df_["loc"] = df_["combination"].apply(lambda x: "VCVS" if "SC_" in x else "Cortex")
df_["hemisphere"] = df_["combination"].apply(lambda x: "Left" if "left" in x else "Right")

df_.to_csv("spectra_all_subjects.csv", index=False)

# loc = "VCVS"
# hem = "Left"
# df_q = df_.query("loc == @loc and hemisphere == @hem and subject == 'aDBS012' and combination == 'SC_0_left'")
# plt.figure(figsize=(10, 15))

# n_lines = df_q.shape[0]
# np.random.seed(42)  # Optional: for reproducibility

# for row in range(n_lines):
#     color = np.random.rand(3,)  # RGB tuple in [0, 1]
#     label = f"Index {str(df_q.iloc[row]['file']).split('_')[2]}"
#     index = df_q.iloc[row]['file'].split('_')[2]
#     if index != "20231115125229383884":  # 20221004114259618432
#         continue
#     plt.plot(df_q.iloc[row, :-6].values, alpha=0.8, color=color, label=label)

# plt.title(f"{loc} - {hem} - aDBS012")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("PSD")
# plt.legend()
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig(f"spectra_{loc}_{hem}_aDBS012.pdf")
# plt.close()


plt_cnt = 0
plt.figure(figsize=(10, 15))  # 8 x 4 plot
for sub in sorted(df_["subject"].unique()):
    for loc_idx, loc in enumerate(df_["loc"].unique()):
        for hem_idx, hemi in enumerate(df_["hemisphere"].unique()):
            plt_cnt += 1
            plt.subplot(8, 4, plt_cnt)
            plt.title(f"{loc} - {hemi}")
            df_sub = df_[(df_["subject"] == sub) & (df_["loc"] == loc) & (df_["hemisphere"] == hemi)]
            if df_sub.empty:
                continue
            for row in range(df_sub.shape[0]):
                plt.plot(df_sub.iloc[row, :-6].values, color="black", alpha=0.3)
            
            plt.xlabel("Frequency (Hz)")
            if loc_idx == 0 and hem_idx == 0:
                plt.ylabel(sub)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/spectra_all_subjects.pdf")
plt.close()

plt.figure(figsize=(8, 6))

plt_cnt = 0
plt.figure(figsize=(6, 6))  # 2 x 3 plot
for loc_idx, loc in enumerate(df_["loc"].unique()):
    for hem_idx, hemi in enumerate(df_["hemisphere"].unique()):
        plt_cnt += 1
        plt.subplot(2, 2, plt_cnt)
        plt.title(f"{loc} - {hemi}")
        d_subs = []
        for sub in sorted(df_["subject"].unique()):
            
            df_sub = df_[(df_["subject"] == sub) & (df_["loc"] == loc) & (df_["hemisphere"] == hemi)]
            if df_sub.empty:
                continue
            #for row in range(df_sub.shape[0]):
            #    plt.plot(df_sub.iloc[row, :-6].values, color="black", alpha=0.3)
            df_sub_mean = df_sub.iloc[:, :-6].mean().values
            plt.plot(np.arange(df_sub_mean.shape[0]), df_sub_mean, color="black", linewidth=0.5, alpha=0.5)
            d_subs.append(df_sub_mean)
        plt.plot(np.arange(df_sub_mean.shape[0]), np.mean(d_subs, axis=0), color="black", linewidth=2)
        plt.xlabel("Frequency (Hz)")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/spectra_all_subjects_combined.pdf")
plt.close()