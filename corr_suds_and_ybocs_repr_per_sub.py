import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'axes.unicode_minus': False
})

from matplotlib import pyplot as plt
import seaborn as sns




feature_names = [
                 "delta",
                 "theta",
                 "alpha",
                 "beta",
                 "gamma",
                 "offset",
                 "exponent",
                 "raw",
                 "RawHjorth_Activity",
                 "RawHjorth_Complexity",
                 "RawHjorth_Mobility",
                 "Sharpwave_Max_prominence_range_1_12",
                 "Sharpwave_Max_prominence_range_1_5",
                 "Sharpwave_Max_sharpness_range_1_12",
                 "Sharpwave_Max_sharpness_range_1_5",
                 "Sharpwave_Mean_interval_range_1_12",
                 "Sharpwave_Mean_interval_range_1_5",
                 "burst_amplitude_delta",
                 "burst_amplitude_theta",
                 "burst_amplitude_alpha",
                 "burst_amplitude_beta",
                 "burst_amplitude_gamma",
                 "burst_duration_alpha_ms",
                 "burst_duration_beta_ms",
                 "burst_duration_delta_ms",
                 "burst_duration_theta_ms",
                 "burst_duration_gamma_ms",
                 ]

regions = ["SC_L", "SC_R", "C_L_1", "C_L_2", "C_R_1", "C_R_2"] 

subs_ = ["004", "005", "007", "008", "009", "010", "011", "012"]
subs_suds = ["004", "005", "007", "009", "010", "011", "012"] # no SUDS for 012

#PATH_ = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/correlation_SUDS_coefficients_patient_ind.npy"
PATH_ = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/correlation_SUDS_coefficients_patient_ind_all_features.npy"

arr_suds_orig = np.load(PATH_)
# dimensions: regions x features x subjects

#arr_ybocs = np.load("corr_score_region_feature_sub.npy")
arr_ybocs_orig = np.load("corr_score_region_feature_sub_all_features.npy")[0, :, :, :]
# dimensions: scores x regions x features x subjects (score (0): YBOCS, )
# delete the 008 subject from arr_ybocs to match arr_suds
arr_ybocs_orig = np.delete(arr_ybocs_orig, 3, axis=2)

plt.figure(figsize=(10, 4))
plt.subplot(121)
rho, p = stats.pearsonr(arr_ybocs_orig[:2, :, :].flatten(), arr_suds_orig[:2, :, :].flatten())
sns.regplot(x=arr_ybocs_orig[:2, :, :].flatten(), y=arr_suds_orig[:2, :, :].flatten(), scatter_kws={"alpha": 0.3})
plt.title(f"VCVS\nr={rho:.3f}, p={p:.4f}")
plt.xlabel("Y-BOCS-II correlation"); plt.ylabel("SUDS correlation")
sns.despine()
plt.subplot(122)
rho, p = stats.pearsonr(arr_ybocs_orig[2:, :, 4:].flatten(), arr_suds_orig[2:, :, 4:].flatten())
sns.regplot(x=arr_ybocs_orig[2:, :, :].flatten(), y=arr_suds_orig[2:, :, :].flatten(), scatter_kws={"alpha": 0.3})
plt.title(f"OFC\nr={rho:.3f}, p={p:.4f}")
plt.xlabel("Y-BOCS-II correlation"); plt.ylabel("SUDS correlation")
sns.despine()
plt.savefig("figures/corr_ybocs_suds_all_features_overall.pdf")

COMPARE_ONLY_SC = False
COMPARE_ONLY_C = True
COMPARE_ONLY_LEFT = False  # both false for comparison with both
COMPARE_ONLY_RIGHT = False
idx_left = [0, 2, 3]
idx_right = [1, 4, 5]

# run a corr for each subject between the two arrays
plt.figure(figsize=(22, 12))

for only_sc in [True, False]:
    if only_sc:
        COMPARE_ONLY_SC = True
        COMPARE_ONLY_C = False
    else:
        COMPARE_ONLY_SC = False
        COMPARE_ONLY_C = True
    for left in [True, False]:
        if left:
            COMPARE_ONLY_LEFT = True
            COMPARE_ONLY_RIGHT = False
        else:
            COMPARE_ONLY_LEFT = False
            COMPARE_ONLY_RIGHT = True

        add_ybocs = []
        add_suds = []
        cnt_ecog = 0
        subs_process = ["004", "005", "007", "009", "010", "011", "012"] # no 012

        for i, sub in enumerate(subs_process):
            if COMPARE_ONLY_C and sub in ["004", "005", "007"]:
                continue
            cnt_offset = 0
            if only_sc is False:
                cnt_offset = 16
            if left is False:
                cnt_offset += 8
            
            #if COMPARE_ONLY_SC:
            plt.subplot(4, 8, i+1 + cnt_offset)
            #else:
            #    plt.subplot(1, 5, cnt_ecog+1)
            #    cnt_ecog += 1
            sub_idx = subs_.index(sub)

            if COMPARE_ONLY_LEFT:
                arr_ybocs = arr_ybocs_orig[idx_left, :, :]
                arr_suds = arr_ybocs_orig[idx_left, :, :]
            elif COMPARE_ONLY_RIGHT:
                arr_ybocs = arr_ybocs_orig[idx_right, :, :]
                arr_suds = arr_suds_orig[idx_right, :, :]
            else:
                arr_ybocs = arr_ybocs_orig
                arr_suds = arr_suds_orig

            if COMPARE_ONLY_SC:
                arr_sub_ybocs = arr_ybocs[:1, :, sub_idx].flatten()
                arr_suds_sub = arr_suds[:1, :, i].flatten()
            elif COMPARE_ONLY_C:
                arr_sub_ybocs = arr_ybocs[1:, :, sub_idx].flatten()
                arr_suds_sub = arr_suds[1:, :, i].flatten()
            else:
                arr_sub_ybocs = arr_ybocs[:, :, sub_idx].flatten()
                arr_suds_sub = arr_suds[:, :, i].flatten() # only SUDS

            mask = ~np.isnan(arr_sub_ybocs) & ~np.isnan(arr_suds_sub)
            add_ybocs.append(arr_sub_ybocs[mask])
            add_suds.append(arr_suds_sub[mask])
            corr, p_value = stats.pearsonr(arr_sub_ybocs[mask], arr_suds_sub[mask])
            # use sns regplot
            sns.regplot(x=arr_sub_ybocs[mask], y=arr_suds_sub[mask], label=f"r={corr:.2f}\np={p_value:.3f}")
            plt.legend()
            #plt.xlabel("YBOCS correlation coefficients")
            if COMPARE_ONLY_SC and COMPARE_ONLY_LEFT:
                plt.title(f"{sub}")
            #plt.ylabel("SUDS correlation coefficients")
            sns.despine()
            #plt.xlim([-0.6, 0.6])
            #plt.ylim([-1, 1])
            if i != 0:
                # remove y ticks and labels
                #plt.yticks([])
                plt.ylabel("")
                # keep the ticks but remove the labels
                #plt.gca().set_yticklabels([])

plt.tight_layout()
if COMPARE_ONLY_SC:
    plt.subplot(1, 8, 8)
elif COMPARE_ONLY_C:
    plt.subplot(1, 5, 5)
# plot across all subjects
#plt.ylim([-1, 1])
arr_sub_ybocs_all = np.concatenate(add_ybocs)
arr_suds_all = np.concatenate(add_suds)
corr, p_value = stats.pearsonr(arr_sub_ybocs_all, arr_suds_all)
sns.regplot(x=arr_sub_ybocs_all, y=arr_suds_all, label=f"r={corr:.2f}\np={p_value:.3f}", scatter_kws={'alpha': 0.3})
plt.legend()
sns.despine()

if COMPARE_ONLY_LEFT:
    str_add = "_left"
elif COMPARE_ONLY_RIGHT:
    str_add = "_right"
else:
    str_add = ""
if COMPARE_ONLY_SC:
    title_ = "Correlation of feature and region correlations\nYBOCS vs SUDS per subject (only SC)"
elif COMPARE_ONLY_C:
    title_ = "Correlation of feature and region correlations\nYBOCS vs SUDS per subject (only C)"
else:
    title_ = "Correlation of feature and region correlations\nYBOCS vs SUDS per subject"

title_ += str_add

plt.suptitle(title_)

if COMPARE_ONLY_SC:
    plt.savefig(f"figures/corr_ybocs_suds_per_sub_only_sc_all_features{str_add}.pdf")
elif COMPARE_ONLY_C:
    plt.savefig(f"figures/corr_ybocs_suds_per_sub_only_c_all_features{str_add}.pdf")
else:
    plt.savefig(f"figures/corr_ybocs_suds_per_sub_all_features{str_add}.pdf")

print("done")


df_res_region_subs = []

plt.figure(figsize=(22, 12))
# rows regions, cols subjects, skip C_ regions if only SC
for idx_region, region in enumerate(regions):
    for idx_sub, sub in enumerate(subs_suds):
        if region in ["C_L_1", "C_L_2", "C_R_1", "C_R_2"] and sub in ["004", "005", "007"]:
            continue  # these subjects don't have cortical electrodes
        plt.subplot(len(regions), len(subs_suds) + 1, idx_region * (len(subs_suds) + 1) + idx_sub + 1)
        arr_ybocs_sub = arr_ybocs_orig[idx_region, :, idx_sub].flatten()
        arr_suds_sub =  arr_suds_orig[idx_region, :, idx_sub].flatten()
        mask = ~np.isnan(arr_ybocs_sub) & ~np.isnan(arr_suds_sub)
        corr, p_value = stats.pearsonr(arr_ybocs_sub[mask], arr_suds_sub[mask])
        # use sns regplot
        sns.regplot(x=arr_ybocs_sub[mask], y=arr_suds_sub[mask], label=f"r={corr:.2f}\np={p_value:.3f}")
        plt.legend()
        df_res_region_subs.append({
            "subject" : sub,
            "region" : region,
            "p" : p_value,
            "corr" : corr,
        })
        
        #plt.xlabel("YBOCS correlation coefficients")
        if idx_region == 0:
            plt.title(f"{sub}")
        #plt.ylabel("SUDS correlation coefficients")
        sns.despine()
        #plt.xlim([-0.6, 0.6])
        #plt.ylim([-1, 1])
        if idx_sub != 0:
            # remove y ticks and labels
            #plt.yticks([])
            plt.ylabel("")
            # keep the ticks but remove the labels
            #plt.gca().set_yticklabels([])
    # plot across all subjects
    plt.subplot(len(regions), len(subs_suds) + 1, (idx_region + 1) * (len(subs_suds) + 1))
    arr_ybocs_all = arr_ybocs_orig[idx_region, :, :].flatten()
    arr_suds_all = arr_suds_orig[idx_region, :, :].flatten()
    mask = ~np.isnan(arr_ybocs_all) & ~np.isnan(arr_suds_all)
    corr, p_value = stats.pearsonr(arr_ybocs_all[mask], arr_suds_all[mask])
    sns.regplot(x=arr_ybocs_all[mask], y=arr_suds_all[mask], label=f"r={corr:.2f}\np={p_value:.3f}", scatter_kws={'alpha': 0.3})
    plt.legend()
    sns.despine()

plt.tight_layout()
plt.savefig("figures/corr_ybocs_suds_per_region_all_features.pdf")

df_res_region_subs = pd.DataFrame(df_res_region_subs)

df_comb = df_res_region_subs.pivot(index="subject", columns="region", values="corr").reset_index()
df_comb = df_comb.astype("float")

plt.figure()
sns.heatmap(data=df_comb.iloc[:, 1:][regions], cmap="coolwarm",  vmin=-0.8, vmax=0.8, annot=False, cbar_kws={"label": "Pearson r"})
plt.yticks(np.arange(len(df_comb["subject"].values)), df_comb["subject"].values, )
# add a star to significant correlations (p < 0.05)
for i, sub in enumerate(df_res_region_subs["subject"].values):
    for j, region in enumerate(regions):
        p = df_res_region_subs[(df_res_region_subs["subject"] == sub) & (df_res_region_subs["region"] == region)]["p"]
        if p.empty:
            continue
        p_val = p.values[0]
        r = df_res_region_subs[(df_res_region_subs["subject"] == sub) & (df_res_region_subs["region"] == region)]["corr"].values[0]
        if p_val < 0.05:
            plt.text(j + 0.5, i + 0.5, f"*", ha="center", va="center", fontsize=12)
        #else:
            #plt.text(j + 0.5, i + 0.5, f"{np.round(r, 2)}", ha="center", va="center", fontsize=12)
plt.savefig("figures/heatmap_corr_ybocs_suds_per_region_all_features.pdf")

