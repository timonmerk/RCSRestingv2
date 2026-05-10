import seaborn as sns
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pickle

score_cols_dict = {
    #'YBOCS_Obs_Comp': ['YBOCS II-Compulsions Sub-score', 'YBOCS II-Obsessions Sub-score'],
    #'YBOCS_Obsessions': ['YBOCS II-Obsessions Sub-score'],
    #'YBOCS_Compulsions': ['YBOCS II-Compulsions Sub-score'],
    'YBOCS_Total': ['YBOCS II Total Score'],
    'BDI': ['BDI-Total Score'],
    'HDRS': ['HDRS Total Score'],
    'BAI': ['BAI-Total Score'],
    'YMRS': ['YMRS Total Score'],
}

arr_label_corrs_only = np.load("figure_3_neural_corr/corr_labels_across_scores.npy")
df_corr_SUDS_YBOCS = pd.read_csv("figure_3_neural_corr/df_corr_SUDS_YBOCS.csv")
corr_SUDS = np.load("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/correlation_SUDS_coefficients.npy")
neural_map_corrs = pd.read_pickle("figure_3_neural_corr/neural_map_corrs.pkl")
ind_score_neural_maps = pd.read_pickle("figure_3_neural_corr/ind_score_neural_maps.pkl")

corr_matrix_coef = neural_map_corrs["arr"]
score_names = neural_map_corrs["scores"]

dict_out = ind_score_neural_maps["dict_out"]
feature_names = ind_score_neural_maps["feature_names"]
regions = ind_score_neural_maps["regions"]

PATH_SAVE = "/Users/Timon/Documents/Houston/resting_state_OCD/figure_3_neural_corr/loc_regions_spectra.pkl"
with open(PATH_SAVE, "rb") as f:
    loc_regions = pickle.load(f)

score_name = "YBOCS_Total"
arr_coef = dict_out["YBOCS_Total"]["arr_coef"]

# PANEL subplot(2, 3, 1)
plt.figure(figsize=(5, 5))
colors = ["#25AB82", "#424186", "#25AB82", "#424186"]
for loc in loc_regions.keys():
    if "SC" in loc:
        plt.subplot(1, 2, 1)
    else:
        plt.subplot(1, 2, 2)
    for spec in loc_regions[loc]:
        plt.plot(spec, color=colors[list(loc_regions.keys()).index(loc)], alpha=0.5)
    plt.plot(np.array(loc_regions[loc]).mean(axis=0), label=f"mean {loc}", color=colors[list(loc_regions.keys()).index(loc)], linewidth=4)
    plt.title(f"{loc} Spectra")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim(0, 90)
    plt.legend()
    plt.ylim(-30, -5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
plt.suptitle("Spectra by Region")
plt.tight_layout()

# SCORE CORRELATIONS
# PANEL subplot(2, 3, 2)
plt.figure(figsize=(5, 4))
sns.heatmap(arr_label_corrs_only, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=list(score_cols_dict.keys()),
            yticklabels=list(score_cols_dict.keys()), cbar_kws={"label": "Pearson corr. coef."},
            vmin=-1, vmax=1)
plt.title("Correlation of Labels across different scores")
plt.xticks(np.arange(len(score_cols_dict)) + 0.5, list(score_cols_dict.keys()), rotation=90)
plt.yticks(np.arange(len(score_cols_dict)) + 0.5, list(score_cols_dict.keys()), rotation=0)
plt.tight_layout()


# PANEL subplot(2, 3, 3)
# SUDS
sns.heatmap(corr_SUDS, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef."},
            vmin=-0.5, vmax=0.5)
plt.tight_layout()

# YBOCS
# PANEL subplot(2, 3, 4)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))#, gridspec_kw={"width_ratios": [0.25, 2]})
sns.heatmap(arr_coef, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, 
            xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef."},
            vmin=-0.5, vmax=0.5)

# PANEL subplot(2, 3, 5)
# correlation of SUDS and YBOCS correlation maps
plt.figure(figsize=(4, 4))
sns.regplot(x=df_corr_SUDS_YBOCS["YBOCS"], y=df_corr_SUDS_YBOCS["SUDS"])
plt.xlabel("YBOCS II Total Score CCA Coefficients")
plt.ylabel("SUDS Correlation Coefficients")
plt.title(f"r = {np.corrcoef(df_corr_SUDS_YBOCS["YBOCS"], df_corr_SUDS_YBOCS["SUDS"])[0,1]:.2f}")
plt.tight_layout()

# PANEL subplot(2, 3, 6)
# matrix of neural maps
plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix_coef, annot=True, fmt=".2f", cmap='coolwarm', 
            cbar_kws={"label": "Pearson corr. coef."},
            vmin=-1, vmax=1, )
plt.xticks(np.arange(len(score_names))+0.5, score_names, rotation=90)
plt.yticks(np.arange(len(score_names))+0.5, score_names, rotation=0)
plt.tight_layout()


# Define main figure with GridSpec (2 rows, 3 cols)
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)   # 2 rows, 3 columns

# ============ PANEL (1,1) with two subplots ============
gs00 = gs[0, 0].subgridspec(1, 2, wspace=0.3)
ax = fig.add_subplot(gs00[0, 0])
loc = "SC_L"
for spec in loc_regions[loc]:
    ax.plot(spec, color=colors[list(loc_regions.keys()).index(loc)], alpha=0.5)
ax.plot(np.array(loc_regions[loc]).mean(axis=0),
        label=f"mean {loc}",
        color=colors[list(loc_regions.keys()).index(loc)],
        linewidth=4)
loc = "SC_R"
for spec in loc_regions[loc]:
    ax.plot(spec, color=colors[list(loc_regions.keys()).index(loc)], alpha=0.5)
ax.plot(np.array(loc_regions[loc]).mean(axis=0),
        label=f"mean {loc}",
        color=colors[list(loc_regions.keys()).index(loc)],
        linewidth=4)
ax.set_title(f"{loc} Spectra")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_xlim(0, 90)
ax.set_ylim(-30, -5)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = fig.add_subplot(gs00[0, 1])
loc = "C_L"
for spec in loc_regions[loc]:
    ax.plot(spec, color=colors[list(loc_regions.keys()).index(loc)], alpha=0.5)
ax.plot(np.array(loc_regions[loc]).mean(axis=0),
        label=f"mean {loc}",
        color=colors[list(loc_regions.keys()).index(loc)],
        linewidth=4)
loc = "C_R"
for spec in loc_regions[loc]:
    ax.plot(spec, color=colors[list(loc_regions.keys()).index(loc)], alpha=0.5)
ax.plot(np.array(loc_regions[loc]).mean(axis=0),
        label=f"mean {loc}",
        color=colors[list(loc_regions.keys()).index(loc)],
        linewidth=4)
ax.set_title(f"{loc} Spectra")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_xlim(0, 90)
ax.set_ylim(-30, -5)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============ PANEL (1,2): Score correlations ============
ax = fig.add_subplot(gs[1, 0])
sns.heatmap(arr_label_corrs_only, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=list(score_cols_dict.keys()),
            yticklabels=list(score_cols_dict.keys()),
            cbar_kws={"label": "Pearson corr. coef."},
            vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation of Labels across different scores")

# ============ PANEL (1,3): SUDS ============
ax = fig.add_subplot(gs[0, 2])
sns.heatmap(corr_SUDS.T[::-1], annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=regions, yticklabels=feature_names[::-1],
            cbar_kws={"label": "Pearson corr. coef."},
            vmin=-0.5, vmax=0.5, ax=ax)
ax.set_title("SUDS")

# ============ PANEL (2,1): YBOCS ============
ax = fig.add_subplot(gs[0, 1])
sns.heatmap(arr_coef.T[::-1], annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=regions, yticklabels=feature_names[::-1],
            cbar_kws={"label": "Pearson corr. coef."},
            vmin=-0.5, vmax=0.5, ax=ax)
ax.set_title("YBOCS")

# ============ PANEL (2,2): Correlation of SUDS & YBOCS maps ============
ax = fig.add_subplot(gs[1, 1])
sns.regplot(x=df_corr_SUDS_YBOCS["YBOCS"],
            y=df_corr_SUDS_YBOCS["SUDS"], ax=ax)
ax.set_xlabel("YBOCS II Total Score CCA Coefficients")
ax.set_ylabel("SUDS Correlation Coefficients")
r_val = np.corrcoef(df_corr_SUDS_YBOCS["YBOCS"],
                    df_corr_SUDS_YBOCS["SUDS"])[0, 1]
p_val = stats.pearsonr(df_corr_SUDS_YBOCS["YBOCS"], df_corr_SUDS_YBOCS["SUDS"])[1]
ax.set_title(f"r = {r_val:.2f}, p = {p_val:.2f}")

# ============ PANEL (2,3): Matrix of neural maps ============
ax = fig.add_subplot(gs[1, 2])
sns.heatmap(corr_matrix_coef, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={"label": "Pearson corr. coef."},
            vmin=-1, vmax=1, ax=ax)
ax.set_xticklabels(score_names, rotation=90)
ax.set_yticklabels(score_names, rotation=0)
ax.set_title("Neural Map Correlation Matrix")

# Final adjustments
plt.suptitle("Summary Figure", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("figure_3_neural_corr/figure_3.pdf")