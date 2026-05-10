import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import cross_decomposition
from matplotlib.backends.backend_pdf import PdfPages

import random
import copy


def permutationTest(x, y, plot_distr=True, x_unit=None, p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distr.
    y (np array) : first distr.
    plot_distr (boolean) : if True: plot permutation histplot and ground truth
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here absolute difference of
    distribution means
    p (float) : p value of permutation test

    """
    # Compute ground truth difference
    gT = np.abs(np.average(x) - np.average(y))

    pV = np.concatenate((x, y), axis=0)
    pS = copy.copy(pV)
    # Initialize permutation:
    pD = []
    # Permutation loop:
    for i in range(0, p):
        # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled
        # distributions and store it in pD:
        pD.append(
            np.abs(
                np.average(pS[0 : int(len(pS) / 2)])
                - np.average(pS[int(len(pS) / 2) :])
            )
        )

    # Calculate p-value
    if gT < 0:
        p_val = len(np.where(pD <= gT)[0]) / p
    else:
        p_val = len(np.where(pD >= gT)[0]) / p

    if plot_distr:
        plt.hist(pD, bins=30, label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth " + x_unit + "=" + str(gT) + " p=" + str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()
    return gT, p_val

df_features = pd.read_csv("features_prep_combined_wide.csv")
col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"


num_subjects = df_features["subject"].nunique()
regions = ["SC_L", "SC_R", "C_L_1", "C_L_2", "C_R_1", "C_R_2"] 
num_regions = len(regions)

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

num_features = len(feature_names)

score_cols_dict = {
    #'YBOCS_Obs_Comp': ['YBOCS II-Compulsions Sub-score', 'YBOCS II-Obsessions Sub-score'],
    #'YBOCS_Obsessions': ['YBOCS II-Obsessions Sub-score'],
    #'YBOCS_Compulsions': ['YBOCS II-Compulsions Sub-score'],
    'YBOCS_Total': 'YBOCS II Total Score',
    'BDI': 'BDI-Total Score',
    'HDRS': 'HDRS Total Score',
    'BAI': 'BAI-Total Score',
    'YMRS': 'YMRS Total Score',
}

dict_out={}
subjects = df_features["subject"].unique()
arr_coef = np.full((len(score_cols_dict.items()), num_regions, num_features, num_subjects), np.nan)
arr_pval = np.full((len(score_cols_dict.items()), num_regions, num_features, num_subjects), np.nan)

for score_name, score_col in score_cols_dict.items():
    score_idx = list(score_cols_dict.keys()).index(score_name)

    dict_out[score_name] = {}
    #for sub_idx, sub in enumerate(df_features["subject"].unique()):
    df_s = df_features.copy() # [df_features["subject"] == sub]

    for region_idx, region in enumerate(regions):
        df_r = df_s[[c for c in df_s.columns if c.startswith(f"{region}_") and "fft_psd" not in c or c == "subject" and "burst" not in c and "Hjorth" not in c and "Sharpwave" not in c and "raw" not in c ]].copy()  # 
        if df_r.empty:
            continue
        df_r = df_r.dropna(axis=1, how='all')
        if df_r.empty:
            continue
        mask = df_r.notna().all(axis=1) & df_s[score_col].notna()
        X = df_r.loc[mask]
        
        for _, sub in enumerate(X["subject"].unique()):
            sub_idx = list(subjects).index(sub)
            mask_sub = X["subject"] == sub
            X_sub = X[mask_sub].copy()

            X_sub = X_sub.drop(columns=["subject"])
            feature_names_sel = [f"{region}_{c}" for c in feature_names]

            X_sub = X_sub[feature_names_sel].values

            Y = df_s[score_col].loc[mask].values
            Y = Y[mask_sub]
            #if len(score_cols) == 1:
            Y = Y.reshape(-1, 1)

            for feature_idx, feature_name in enumerate(feature_names_sel):
                #for j, score_col in enumerate(np.arange(Y.shape[1])):
                corr, p_value = stats.pearsonr(X_sub[:, feature_idx], Y[:, 0])
                arr_coef[score_idx, region_idx, feature_idx, sub_idx] = corr
                arr_pval[score_idx, region_idx, feature_idx, sub_idx] = p_value

        dict_out[score_name]["arr_coef"] = arr_coef

arr_coef = arr_coef[0, :, :, :]
arr_pval = arr_pval[0, :, :, :]

subjects = df_features["subject"].unique()

rows = []

for sub_idx, subject in enumerate(subjects):
    for region_idx, region in enumerate(regions):
        for feature_idx, feature in enumerate(feature_names):
            coef = arr_coef[region_idx, feature_idx, sub_idx]
            pval = arr_pval[region_idx, feature_idx, sub_idx]

            rows.append({
                "subject": subject,
                "region": region,
                "feature": feature,
                "correlation": coef,
                "p_value": pval
            })

df_coefs = pd.DataFrame(rows)

##################################################################################

# Subjects of interest
ecog_subjects = ["aDBS009", "aDBS010", "aDBS011", "aDBS012"]
subjects_all = df_features["subject"].unique()
subj_to_idx = {s: i for i, s in enumerate(subjects_all)}

# Region indices for cortical ECoG contacts
ecog_regions = [2, 3, 4, 5]
ecog_labels  = ["C_L_1", "C_L_2", "C_R_1", "C_R_2"]

n_cols = len(ecog_subjects) + 1
fig, axes = plt.subplots(1, n_cols, figsize=(12, 3), sharey=True)

if n_cols == 1:
    axes = [axes]

for col_idx, sub in enumerate(ecog_subjects):
    sidx = subj_to_idx[sub]
    arr_plot = arr_coef[ecog_regions, :, sidx]   # shape (4, features)
    arr_p    = arr_pval[ecog_regions, :, sidx]

    # Build annotation matrix with stars
    vals  = arr_plot.T    # (features, 4)
    pvals = arr_p.T
    annot = np.empty_like(vals, dtype=object)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            p = pvals[i, j]
            star = "*" if (np.isfinite(p) and p < 0.05) else ""
            annot[i, j] = f"{v:.2f}{star}"

    ax = axes[col_idx]
    sns.heatmap(vals, annot=annot, cmap="coolwarm",
                vmin=-0.5, vmax=0.5,
                yticklabels=feature_names,
                xticklabels=ecog_labels,
                fmt="", cbar=False,#(col_idx==0),
                annot_kws={"ha": "center", "va": "center", "size": 8},
                ax=ax)

    #ax.invert_yaxis()
    ax.set_title(f"Subject {sub}", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    # # Only keep y labels on first subplot
    # if col_idx != 0:
    #     ax.set_ylabel("")
    #     ax.set_yticks([])
    #     ax.set_yticklabels([])

# plot the subject average in the last subplot
arr_plot = np.nanmean(arr_coef[ecog_regions, :, :][:, :, [subj_to_idx[s] for s in ecog_subjects]], axis=2)   # shape (4, features)
#arr_p    = np.nanmean(arr_pval[ecog_regions, :, [subj_to_idx[s] for s in ecog_subjects]], axis=2)   # same shape
# Build annot matrix (features x 2) with "\n*" if p < 0.05
vals   = arr_plot.T                                   # (features, 4)
pvals  = arr_p.T                                      # (features, 4)
annot  = np.empty_like(vals, dtype=object)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        p = pvals[i, j]
        # compute a permutation test p-value for this feature across subjects
        feature_ = feature_names[i]
        region_ = ecog_labels[j]
        corrs_ = df_coefs.query("region == @region_ and feature == @feature_")["correlation"].values[3:]
        p_val = permutationTest(corrs_, np.zeros(corrs_.shape), plot_distr=False, x_unit="correlation", p=5000)[1]
        star = "*" if (isinstance(p_val, (float, np.floating)) and np.isfinite(p_val) and p_val < 0.05) else ""
        annot[i, j] = f"{star}{v:.2f}"
ax = axes[-1]
sns.heatmap(vals, annot=annot, cmap="coolwarm",
            vmin=-0.5, vmax=0.5,
            yticklabels=feature_names,
            xticklabels=ecog_labels,
            #cbar=(col_idx==0),
            fmt="",
            annot_kws={"ha": "center", "va": "center", "size": 8},
            ax=ax)
ax.set_title(f"Average", fontsize=8)
ax.invert_yaxis()
#cbar = axes[-1].collections[0].colorbar
#cbar.ax.set_position([0.92, 0.15, 0.02, 0.7])

# Shared colorbar
# cbar = axes[0].collections[0].colorbar
# cbar.ax.set_position([0.92, 0.15, 0.02, 0.7])

#plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("figures/1709/fig2_corr_ECoG.pdf")
plt.show()


######
plt.figure()
df_plt = df_coefs.query("subject != 'aDBS004' and subject != 'aDBS005' and subject != 'aDBS007'")
df_plt["correlation"] = df_plt["correlation"].abs()
df_plt = df_plt.groupby(["subject", "region"])["correlation"].max().reset_index()
ax = sns.boxplot(x="region", y="correlation", data=df_plt, showfliers=False)
sns.swarmplot(x="region", y="correlation", data=df_plt, color=".25", dodge=True)
# set fontsize to 8
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_title(f"Absolute correlation coeff.", fontsize=8)
plt.savefig("figures/1709/fig2_2_all_regions_correlations_sum.pdf", bbox_inches='tight')

fig, axes = plt.subplots(1, 8, figsize=(12, 3), sharey=True)

for sub_idx, sub in enumerate(subjects[:7]):
    arr_plot = arr_coef[[0, 1], :, sub_idx]              # shape: (2, features)
    arr_p    = arr_pval[[0, 1], :, sub_idx]              # same shape

    # Build annot matrix (features x 2) with "\n*" if p < 0.05
    vals   = arr_plot.T                                   # (features, 2)
    pvals  = arr_p.T                                      # (features, 2)
    annot  = np.empty_like(vals, dtype=object)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            p = pvals[i, j]
            star = "*" if (isinstance(p, (float, np.floating)) and np.isfinite(p) and p < 0.05) else ""
            annot[i, j] = f"{star}{v:.2f}"

    ax = axes[sub_idx]
    sns.heatmap(vals, annot=annot, cmap="coolwarm",
                vmin=-0.5, vmax=0.5,
                yticklabels=feature_names,
                xticklabels=["SC_L", "SC_R"],
                fmt="", cbar=False,
                annot_kws={"ha": "center", "va": "center", "size": 8},
                ax=ax)

    #ax.invert_yaxis()
    ax.set_title(f"Subject {sub}", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# plot the mean across subjects in the last subplot
# plot the subject average in the last subplot
arr_plot = np.nanmean(arr_coef[[0, 1], :, :7], axis=2)   # shape (2, features)
#arr_p    = np.nanmean(arr_pval[[0, 1], :, :7], axis=2)   # same shape
# Build annot matrix (features x 2) with "\n*" if p < 0.05
vals   = arr_plot.T                                   # (features, 2)
pvals  = arr_p.T                                      # (features, 2)
annot  = np.empty_like(vals, dtype=object)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        p = pvals[i, j]
        # compute a permutation test p-value for this feature across subjects
        feature_ = feature_names[i]
        region_ = regions[j]
        corrs_ = df_coefs.query("region == @region_ and feature == @feature_")["correlation"].values
        p_val = permutationTest(corrs_, np.zeros(corrs_.shape), plot_distr=False, x_unit="correlation", p=5000)[1]
        star = "*" if (isinstance(p_val, (float, np.floating)) and np.isfinite(p_val) and p_val < 0.05) else ""
        annot[i, j] = f"{star}{v:.2f}"
ax = axes[7]
sns.heatmap(vals, annot=annot, cmap="coolwarm",
            vmin=-0.5, vmax=0.5,
            yticklabels=feature_names,
            xticklabels=["SC_L", "SC_R"],
            cbar=False,#(sub_idx==0),
            fmt="",
            annot_kws={"ha": "center", "va": "center", "size": 8},
            ax=ax)
ax.set_title(f"Average", fontsize=8)
ax.invert_yaxis()
# single shared colorbar to the right
# cbar = axes[0].collections[0].colorbar
# cbar.ax.set_position([0.92, 0.15, 0.02, 0.7])
#plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("figures/1709/fig2_corr_VCVS.pdf")

##################################################################################


# subset for SC regions only
df_plot = df_coefs.query("region.str.startswith('SC_')")


p_vals = []
for feature in feature_names:
    _, p_val = permutationTest(df_plot.query("feature == @feature and region == 'SC_L'")["correlation"].values - 
                               df_plot.query("feature == @feature and region == 'SC_R'")["correlation"].values,
                                 np.zeros(df_plot.query("feature == @feature and region == 'SC_L'")["correlation"].shape),
                               plot_distr=False, x_unit="correlation", p=5000)
    p_vals.append(p_val)

plt.figure(figsize=(9,3))
plt.subplot(121)
ax = sns.boxplot(x="feature", hue="region", y="correlation", data=df_plot, showfliers=False)
sns.swarmplot(x="feature", hue="region", y="correlation", legend=False,
              data=df_plot, color=".25", dodge=True)

# loop over features and corresponding p-values
features = df_plot["feature"].unique()
y_margin = (df_plot["correlation"].max() - df_plot["correlation"].min()) * 0.05

for i, (feat, p_val) in enumerate(zip(features, p_vals)):
    # find the max correlation for this feature
    y_max = df_plot.loc[df_plot["feature"] == feat, "correlation"].max()
    y_pos = y_max + y_margin
    # annotate above the tallest box/swarm for that feature
    ax.text(i, y_pos, f"p={p_val:.3f}", ha="center", va="bottom", fontsize=8)

plt.ylabel("Pearson r")
ax.set_title(f"Pearson orrelation coeff.", fontsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
# turn off upper and right spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

df_plot_abs = df_plot.copy()
df_plot_abs["correlation"] = df_plot_abs["correlation"].abs()
p_vals = []
for feature in feature_names:
    _, p_val = permutationTest(df_plot_abs.query("feature == @feature and region == 'SC_L'")["correlation"].values -
                               df_plot_abs.query("feature == @feature and region == 'SC_R'")["correlation"].values,
                               np.zeros(df_plot_abs.query("feature == @feature and region == 'SC_L'")["correlation"].shape),
                               plot_distr=False, x_unit="correlation", p=5000)
    p_vals.append(p_val)

plt.subplot(122)
ax = sns.boxplot(x="feature", hue="region", y="correlation", data=df_plot_abs, showfliers=False)
sns.swarmplot(x="feature", hue="region", y="correlation", legend=False,
              data=df_plot_abs, color=".25", dodge=True)

# loop over features and corresponding p-values
features = df_plot_abs["feature"].unique()
y_margin = (df_plot_abs["correlation"].max() - df_plot_abs["correlation"].min()) * 0.05

for i, (feat, p_val) in enumerate(zip(features, p_vals)):
    # find the max correlation for this feature
    y_max = df_plot_abs.loc[df_plot_abs["feature"] == feat, "correlation"].max()
    y_pos = y_max + y_margin
    # annotate above the tallest box/swarm for that feature
    ax.text(i, y_pos, f"p={p_val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_title(f"Absolute correlation coeff.", fontsize=8)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
# turn off upper and right spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.savefig("figures/1709//fig2_2_sc_lr_correlations.pdf", bbox_inches='tight')

plt.show()

## ok, now the next plot with ecog comparison 
plt.figure(figsize=(14, 5))
plt.subplot(121)
df_plt = df_coefs.query("subject != 4 and subject != 5 and subject != 7")
ax = sns.boxplot(x="feature", hue="region", y="correlation", data=df_plt, showfliers=False)
sns.swarmplot(x="feature", hue="region", y="correlation", legend=False,
              data=df_plt, color=".25", dodge=True)
# set the fontsizes to 8
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_title(f"Pearson correlation coeff.", fontsize=8)

plt.subplot(122)
df_plt_ = df_coefs.query("subject != 4 and subject != 5 and subject != 7")
df_plt_["correlation"] = df_plt_["correlation"].abs()
ax = sns.boxplot(x="feature", hue="region", y="correlation", data=df_plt_, showfliers=False)
sns.swarmplot(x="feature", hue="region", y="correlation", legend=False,
              data=df_plt_, color=".25", dodge=True)
# set the fontsizes to 8
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_title(f"Absolute correlation coeff.", fontsize=8)
plt.savefig("figures/fig2_2_all_regions_correlations.pdf", bbox_inches='tight')


# now make a figure where the regions are summed up for all frequency bands
plt.figure(figsize=(9, 3))
plt.subplot(121)
df_plt = df_coefs.query("subject != 'aDBS004' and subject != 'aDBS005' and subject != 'aDBS007'")
ax = sns.boxplot(x="region", y="correlation", data=df_plt, showfliers=False)
# set fontsize to 8
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_title(f"Pearson correlation coeff.", fontsize=8)

sns.swarmplot(x="region", y="correlation", data=df_plt, color=".25", dodge=True)
plt.subplot(122)
df_plt_ = df_coefs.query("subject != 'aDBS004' and subject != 'aDBS005' and subject != 'aDBS007'")
df_plt_["correlation"] = df_plt_["correlation"].abs()
ax = sns.boxplot(x="region", y="correlation", data=df_plt_, showfliers=False)
sns.swarmplot(x="region", y="correlation", data=df_plt_, color=".25", dodge=True)
# set fontsize to 8
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.set_title(f"Absolute correlation coeff.", fontsize=8)
plt.savefig("figures/1709/fig2_2_all_regions_correlations_sum.pdf", bbox_inches='tight')



############################################################################





# save arr_coef
#np.save("corr_score_region_feature_sub.npy", arr_coef)
np.save("corr_score_region_feature_sub_all_features.npy", arr_coef)

# score_idx, region_idx, feature_idx, sub_idx
# make a pdf plot with patient indiviedual maps for each region vs feature

pdf_ = PdfPages("figures/individual_corr_maps.pdf")
for score_name, score_cols in score_cols_dict.items():
    plt.figure(figsize=(15, 10))
    for sub_idx, sub in enumerate(df_features["subject"].unique()):
        plt.subplot(3, 4, sub_idx+1)
        arr_plt = arr_coef[list(score_cols_dict.keys()).index(score_name), :, :, sub_idx]
        sns.heatmap(arr_plt, annot=True, fmt=".2f", cmap='coolwarm', 
                    xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef."},
                    annot_kws={"size": 7},
                    vmin=-0.5, vmax=0.5)
        plt.title(f"{sub}")
    plt.suptitle(f"{score_name}")
    plt.tight_layout()
    pdf_.savefig(plt.gcf())
    plt.close()
pdf_.close()





# ok, plot the average now across subjects, and in a separate plot the std across subjects
pdf_ = PdfPages("figures/average_corr_maps_0409.pdf")
for score_name, score_cols in score_cols_dict.items():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    arr_plt = np.nanmean(arr_coef[list(score_cols_dict.keys()).index(score_name), :, :, :], axis=-1)
    sns.heatmap(arr_plt, annot=True, fmt=".2f", cmap='coolwarm', 
                xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef."},
                vmin=-0.5, vmax=0.5)
    plt.title(f"Mean Correlation Coefficients for {score_name}")
    plt.subplot(1, 2, 2)
    arr_plt = np.nanstd(arr_coef[list(score_cols_dict.keys()).index(score_name), :, :, :], axis=-1)
    sns.heatmap(arr_plt, annot=True, fmt=".2f", cmap='coolwarm', 
                xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef. std"},
                vmin=0, vmax=0.5)
    plt.title(f"Std of Correlation Coefficients for {score_name}")
    plt.tight_layout()
    pdf_.savefig(plt.gcf())
    plt.close()
pdf_.close()

# correlate now each arr_coef matrix for each score
corr_SUDS = np.load("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/correlation_SUDS_coefficients.npy")

corr_matrix_coef = np.zeros([len(score_cols_dict.values())+1, len(score_cols_dict.values())+1])

for i, (score_name_i, score_cols_i) in enumerate(score_cols_dict.items()):
    for j, (score_name_j, score_cols_j) in enumerate(score_cols_dict.items()):
        arr_coef_i = dict_out[score_name_i]["arr_coef"]
        arr_coef_j = dict_out[score_name_j]["arr_coef"]
        corr_matrix_coef[i, j] = np.corrcoef(arr_coef_i.flatten(), arr_coef_j.flatten())[0, 1]
for i, (score_name_i, score_cols_i) in enumerate(score_cols_dict.items()):
    corr_matrix_coef[i, -1] = np.corrcoef(dict_out[score_name_i]["arr_coef"].flatten(), corr_SUDS.flatten())[0, 1]
    corr_matrix_coef[-1, i] = corr_matrix_coef[i, -1]
corr_matrix_coef[-1, -1] = 1

score_corrs_dict = {
    "arr" : arr_label_corrs_only,
    "scores": list(score_cols_dict.keys())
}
pd.to_pickle(score_corrs_dict, "figure_3_neural_corr/score_corrs_dict.pkl")

s1_ybocs = dict_out["YBOCS_Total"]["arr_coef"].flatten()
s2_suds = corr_SUDS.flatten()
df_corr_SUDS_YBOCS = pd.DataFrame({"YBOCS": s1_ybocs, "SUDS": s2_suds})
df_corr_SUDS_YBOCS.to_csv("figure_3_neural_corr/df_corr_SUDS_YBOCS.csv", index=False)
# make a regplot
plt.figure(figsize=(4, 4))
sns.regplot(x=s1_ybocs, y=s2_suds)
plt.xlabel("YBOCS II Total Score CCA Coefficients")
plt.ylabel("SUDS Correlation Coefficients")
plt.title(f"r = {np.corrcoef(s1_ybocs, s2_suds)[0,1]:.2f}")
plt.tight_layout()



neural_map_corrs = {
    "arr" : corr_matrix_coef,
    "scores": list(score_cols_dict.keys()) + ["SUDS"]
}
pd.to_pickle(neural_map_corrs, "figure_3_neural_corr/neural_map_corrs.pkl")

ind_score_neural_maps = {
    "dict_out": dict_out,
    "feature_names": feature_names,
    "regions": regions,
}
pd.to_pickle(ind_score_neural_maps, "figure_3_neural_corr/ind_score_neural_maps.pkl")

plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix_coef, annot=True, fmt=".2f", cmap='coolwarm', 
            xticklabels=list(score_cols_dict.keys()),
            yticklabels=list(score_cols_dict.keys()), cbar_kws={"label": "Pearson corr. coef."},
            vmin=-1, vmax=1, )
if GET_CORRELATIONS:
    plt.title("Correlation across corr maps")
else:
    plt.title("Correlation of CCA coefficients across different scores")
plt.xticks(np.arange(len(score_cols_dict)+1)+0.5, list(score_cols_dict.keys())+["SUDS"], rotation=90)
plt.yticks(np.arange(len(score_cols_dict)+1)+0.5, list(score_cols_dict.keys())+["SUDS"], rotation=0)
plt.tight_layout()
if GET_CORRELATIONS:
    plt.savefig("figures/corr_corrmaps.pdf")
else:
    plt.savefig("figures/cca_coef_correlations_different_scores.pdf")


# plot now arr_label_corrs_only
plt.figure(figsize=(5, 4))
sns.heatmap(arr_label_corrs_only, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=list(score_cols_dict.keys()),
            yticklabels=list(score_cols_dict.keys()), cbar_kws={"label": "Pearson corr. coef."},
            vmin=-1, vmax=1)
plt.title("Correlation of Labels across different scores")
plt.xticks(np.arange(len(score_cols_dict)) + 0.5, list(score_cols_dict.keys()), rotation=90)
plt.yticks(np.arange(len(score_cols_dict)) + 0.5, list(score_cols_dict.keys()), rotation=0)
plt.tight_layout()
plt.savefig("figures/cca_label_correlations_scores_only.pdf")


# create a pdf with all scores as individual figures
# show on the left subplot the arr_coef_labels and on the right subplot the arr_coef
pdf_name = "figures/corr_individual_maps.pdf" if GET_CORRELATIONS else "figures/cca_coef_individual_scores_cca.pdf"
with PdfPages(pdf_name) as pdf:
    for score_name, score_data in dict_out.items():
        arr_coef = score_data["arr_coef"]
        arr_coef_labels = score_data["arr_coef_labels"]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))#, gridspec_kw={"width_ratios": [0.25, 2]})
        
        # Left subplot: arr_coef_labels, should be a barplot
        #sns.heatmap(arr_coef_labels, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[0])
        # sns.barplot(x=np.arange(len(score_cols_dict[score_name])), y=arr_coef_labels.mean(axis=0), ax=axes[0])
        # if GET_CORRELATIONS:
        #     axes[0].set_title(f"Correlation Labels for {score_name}")
        # else:
        #     axes[0].set_title(f"CCA Coefficients Labels for {score_name}")
        # axes[0].set_xlabel("Subjects and Regions")
        # #axes[0].set_yticks(np.arange(len(regions))+0.5)
        # #axes[0].set_yticklabels(regions, rotation=0)
        # axes[0].set_xticks(np.arange(len(score_cols_dict[score_name])))
        # axes[0].set_xticklabels(score_cols_dict[score_name], rotation=90)

        # Right subplot: arr_coef
        sns.heatmap(arr_coef, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, 
                   xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef."},
                   vmin=-0.5, vmax=0.5)
        if GET_CORRELATIONS:
            ax.set_title(f"Correlation Coefficients for {score_name}")
        else:
            ax.set_title(f"CCA Coefficients for {score_name}")
        # ax.set_yticks(np.arange(len(regions))+0.5)
        # ax.set_yticklabels(regions, rotation=0)
        # ax.set_xticks(np.arange(len(feature_names))+0.5)
        # ax.set_xticklabels(feature_names, rotation=90)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    # add the figure now for SUDS
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_SUDS, annot=True, fmt=".2f", cmap='coolwarm', ax=ax,
                xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef."},
                vmin=-0.5, vmax=0.5)
    ax.set_title("Correlation of SUDS Coefficients")
    # ax.set_yticks(np.arange(len(regions))+0.5)
    # ax.set_yticklabels(regions, rotation=0)
    # ax.set_xticks(np.arange(len(feature_names))+0.5)
    # ax.set_xticklabels(feature_names, rotation=90)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print()