import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import cross_decomposition
from matplotlib.backends.backend_pdf import PdfPages

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
                #  "raw",
                #   "RawHjorth_Activity",
                #   "RawHjorth_Complexity",
                #   "RawHjorth_Mobility",
                #   "Sharpwave_Max_prominence_range_1_12",
                #   "Sharpwave_Max_prominence_range_1_5",
                #   "Sharpwave_Max_sharpness_range_1_12",
                #   "Sharpwave_Max_sharpness_range_1_5",
                #   "Sharpwave_Mean_interval_range_1_12",
                #   "Sharpwave_Mean_interval_range_1_5",
                #   "burst_amplitude_delta",
                #   "burst_amplitude_theta",
                #   "burst_amplitude_alpha",
                #   "burst_amplitude_beta",
                #   "burst_amplitude_gamma",
                #   "burst_duration_alpha_ms",
                #   "burst_duration_beta_ms",
                #   "burst_duration_delta_ms",
                #   "burst_duration_theta_ms",
                #   "burst_duration_gamma_ms",
                 ]

num_features = len(feature_names)

# score_col = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
# score_cols = ['YBOCS II Total Score', 'YBOCS II-Obsessions Sub-score', 'YBOCS II-Compulsions Sub-score']
# score_cols = ["YMRS Total Score"]
# score_cols = ['BDI-Total Score', 'HDRS Total Score']
# score_cols = ['BAI-Total Score', ]

score_cols_dict = {
    #'YBOCS_Obs_Comp': ['YBOCS II-Compulsions Sub-score', 'YBOCS II-Obsessions Sub-score'],
    'YBOCS_Obsessions': ['YBOCS II-Obsessions Sub-score'],
    'YBOCS_Compulsions': ['YBOCS II-Compulsions Sub-score'],
    'YBOCS_Total': ['YBOCS II Total Score'],
    'BDI': ['BDI-Total Score'],
    'HDRS': ['HDRS Total Score'],
    'BAI': ['BAI-Total Score'],
    'YMRS': ['YMRS Total Score'],
}

NORMALIZE_WITHIN_SUBJECTS = False
NORMALIZE_BEFORE_CCA = True
GET_CORRELATIONS = False

dict_out={}

for score_name, score_cols in score_cols_dict.items():
    if score_name == "BDI_HDRS":
        print("Here")
    arr_coef = np.full((num_regions, num_features), np.nan)
    arr_coef_labels = np.full((num_regions, len(score_cols)), np.nan)
    dict_out[score_name] = {}
    #for sub_idx, sub in enumerate(df_features["subject"].unique()):
    df_s = df_features.copy() # [df_features["subject"] == sub]
        
    for region_idx, region in enumerate(regions):
        #if (sub == "004" or sub == "005" or sub == "007") and region.startswith("C_"):
        #    continue

        df_r = df_s[[c for c in df_s.columns if c.startswith(f"{region}_") and "fft_psd" not in c or c == "subject" and "burst" not in c and "Hjorth" not in c and "Sharpwave" not in c and "raw" not in c ]].copy()  # 
        if df_r.empty:
            continue
        df_r = df_r.dropna(axis=1, how='all')
        if df_r.empty:
            continue
        mask = df_r.notna().all(axis=1) & df_s[score_cols].notna().all(axis=1)
        X = df_r.loc[mask]
        if NORMALIZE_WITHIN_SUBJECTS:

            # normalize for each individual subject each column
            X = X.groupby("subject").transform(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1))
        # drop subject column
        else:
            X = X.drop(columns=["subject"])
        feature_names_sel = [f"{region}_{c}" for c in feature_names]

        X = X[feature_names_sel].values

        Y = df_s[score_cols].loc[mask].values
        if len(score_cols) == 1:
            Y = Y.reshape(-1, 1)

        if NORMALIZE_BEFORE_CCA:
            X = StandardScaler().fit_transform(X)
            Y = StandardScaler(with_mean=True, with_std=True).fit_transform(Y)
        if GET_CORRELATIONS is True:
            for i, score_col in enumerate(np.arange(Y.shape[1])):
                corr_vals = []
                for j, score_col_2 in enumerate(np.arange(Y.shape[1])):
                    if i == j:
                        continue
                    corr, p_value = stats.pearsonr(Y[:, i], Y[:, j])
                    corr_vals.append(corr)
                if len(corr_vals) == 0:
                    corr_vals = [1]
                arr_coef_labels[region_idx, i] = np.mean(corr_vals)
            for i, feature_name in enumerate(feature_names_sel):
                corr_vals = []
                for j, score_col in enumerate(np.arange(Y.shape[1])):
                    corr, p_value = stats.pearsonr(X[:, i], Y[:, int(score_col)])
                    corr_vals.append(corr)
                arr_coef[region_idx, i] = np.mean(corr_vals)
        else:
            cca = cross_decomposition.CCA(n_components=1, max_iter=5000, tol=1e-06, scale=False)
            cca.fit(X, Y)

            cca_x_weight = cca.x_weights_
            cca_y_weight = cca.y_weights_
            if cca_y_weight[0, 0] < 0:
                cca_x_weight = -cca_x_weight  # ensure that the first dimension is always positive
                cca_y_weight = -cca_y_weight

            arr_coef[region_idx, :] = cca_x_weight.ravel()
            arr_coef_labels[region_idx, :] = cca_y_weight.ravel()
    dict_out[score_name]["arr_coef"] = arr_coef
    dict_out[score_name]["arr_coef_labels"] = arr_coef_labels

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

# plot
plt.figure(figsize=(8, 6))
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