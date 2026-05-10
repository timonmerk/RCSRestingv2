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

cols_test = ["YBOCS II Total Score",
             'YBOCS II-Obsessions Sub-score',
             'YBOCS II-Compulsions Sub-score',
             'Category 1:Concerns about Germs and Cotamination- Subscale Score',
             'Category 2: Concerns about being Responsible for Harm, Injury, or Bad Luck- Subscale Scale',
             'Category 3: Unacceptable Thoughts-Subscale Score',
             'Category 4: Concerns about Symmetry, Completeness',
             #'Category 5: Sexually Intrusive Thoughts- Subscale Score',
             #'Category 6: Intrusive Violent Thoughts- Subscale Score',
             #'Category 7: Immoral and Scrupulous Thoughts- Subscale Score', 
             'BDI-Total Score',
             'HDRS Total Score',
             'BAI-Total Score',
             'YMRS Total Score',
]

cols_labels = [
    "YBOCS II Total",
    "YBOCS II Obs",
    "YBOCS II Comp",
    "DOCS 1 Contamination",
    "DOCS 2 Responsibility",
    "DOCS 3 Unacceptable Thoughts",
    "DOCS 4 Symmetry",
    "BDI",
    "HDRS",
    "BAI",
    "YMRS",
]

max_scores = {
    "YBOCS II Total Score": 50,
    'YBOCS II-Obsessions Sub-score': 25,
    'YBOCS II-Compulsions Sub-score': 25,
    'Category 1:Concerns about Germs and Cotamination- Subscale Score': 20,
    "Category 2: Concerns about being Responsible for Harm, Injury, or Bad Luck- Subscale Scale": 20,
    "Category 3: Unacceptable Thoughts-Subscale Score": 20,
    "Category 4: Concerns about Symmetry, Completeness": 20,
    'BDI-Total Score': 63,
    "HDRS Total Score": 52,  # 17-item version
    "BAI-Total Score": 63,
    "YMRS Total Score": 60,
}

dict_out={}
subjects = df_features["subject"].unique()
arr_coef = np.full((len(cols_test), num_regions, num_features, num_subjects), np.nan)

for score_idx, score_name in enumerate(cols_test):

    df_s = df_features.copy() # [df_features["subject"] == sub]
    for region_idx, region in enumerate(regions):
        df_r = df_s[[c for c in df_s.columns if c.startswith(f"{region}_") and "fft_psd" not in c or c == "subject" and "burst" not in c and "Hjorth" not in c and "Sharpwave" not in c and "raw" not in c ]].copy()  # 
        if df_r.empty:
            continue
        df_r = df_r.dropna(axis=1, how='all')
        if df_r.empty:
            continue
        mask = df_r.notna().all(axis=1) & df_s[score_name].notna()
        X = df_r.loc[mask]
        
        for _, sub in enumerate(X["subject"].unique()):
            sub_idx = list(subjects).index(sub)
            mask_sub = X["subject"] == sub
            X_sub = X[mask_sub].copy()

            X_sub = X_sub.drop(columns=["subject"])
            feature_names_sel = [f"{region}_{c}" for c in feature_names]

            X_sub = X_sub[feature_names_sel].values

            Y = df_s[score_name].loc[mask].values
            Y = Y[mask_sub]
            #if len(score_cols) == 1:
            Y = Y.reshape(-1, 1)

            for feature_idx, feature_name in enumerate(feature_names_sel):
                #for j, score_col in enumerate(np.arange(Y.shape[1])):
                corr, p_value = stats.pearsonr(X_sub[:, feature_idx], Y[:, 0])
                arr_coef[score_idx, region_idx, feature_idx, sub_idx] = corr

def get_img_scores(SUBJECT_PLT):
    img_scores = df_features.query("subject == @SUBJECT_PLT")[cols_test + ["date"]]
    img_scores["date"] = pd.to_datetime(img_scores["date"])
    first_date = img_scores["date"].min()
    img_scores["days_since_first"] = (img_scores["date"] - first_date).dt.days
    img_scores = img_scores.set_index("days_since_first")
    img_scores = img_scores.drop(columns=["date"])
    # normalize by max score
    for col in cols_test:
        if col in max_scores:
            img_scores[col] = img_scores[col] / max_scores[col] * 100
    return img_scores

def get_img_SC_L(SUBJECT_PLT):
    map_spec_region = df_features.query("subject == @SUBJECT_PLT")
    cols_regions = [f"SC_L_{c}" for c in feature_names]
    map_spec_region = map_spec_region[cols_regions + ["date"]]
    date_map_spec = pd.to_datetime(map_spec_region["date"])
    first_date = date_map_spec.min()
    days_since_first_map = (date_map_spec - first_date).dt.days
    scaler = StandardScaler()
    map_spec_region[cols_regions] = scaler.fit_transform(map_spec_region[cols_regions])
    return map_spec_region, days_since_first_map, cols_regions

def get_img_delta_power(SUBJECT_PLT):
    map_spec_features = df_features.query("subject == @SUBJECT_PLT")
    cols_features = [c for c in map_spec_features.columns if c.endswith("_delta") and "fft" not in c and "Unknown" not in c and "Misc" not in c and "burst" not in c or c == "date"]
    map_spec_features = map_spec_features[cols_features]
    date_map_feature = pd.to_datetime(map_spec_features["date"])
    first_date = date_map_feature.min()
    days_since_first_map_feature = (date_map_feature - first_date).dt.days
    map_spec_features = map_spec_features.drop(columns=["date"])
    cols_features = [c for c in cols_features if c != "date"]
    scaler = StandardScaler()
    map_spec_features[cols_features] = scaler.fit_transform(map_spec_features[cols_features])
    return map_spec_features, days_since_first_map_feature, cols_features


def get_map_yboc_corr(SUBJECT_PLT):
    sub_idx = list(subjects).index(SUBJECT_PLT)
    map_neural_corr_ybocs = arr_coef[0, :, :, sub_idx]  # for YBOCS_Total (idx 0)
    return map_neural_corr_ybocs

def get_label_correlations(SUBJECT_PLT):
    sub_idx = list(subjects).index(SUBJECT_PLT)
    map_symptom_correlations = np.zeros((len(cols_labels), len(cols_labels)))
    for i, label_i in enumerate(cols_labels):
        for j, label_j in enumerate(cols_labels):
            if i != j:
                corr, _ = stats.pearsonr(arr_coef[i, :, :, sub_idx].flatten(), arr_coef[j, :, :, sub_idx].flatten())
                map_symptom_correlations[i, j] = corr
            else:
                map_symptom_correlations[i, j] = 1.0
    return map_symptom_correlations


FREQ = "30D"  # resample bin

# get here the resamples mean values
l_sub_scores = []
for sub in df_features["subject"].unique():
    df_score_sub = get_img_scores(sub)
    df_score_sub.index = pd.to_timedelta(df_score_sub.index, unit="D")
    df_score_sub = df_score_sub.resample('30D').mean()
    df_score_sub = df_score_sub[df_score_sub.index < pd.Timedelta(days=600)]
    df_score_sub["subject"] = sub
    l_sub_scores.append(df_score_sub)
df_scores_resampled = pd.concat(l_sub_scores, ignore_index=False)
# average across subjects
df_scores_resampled_mean = df_scores_resampled[[c for c in df_scores_resampled.columns if c != "subject"]].groupby(df_scores_resampled.index).mean()
df_scores_resampled_std = df_scores_resampled[[c for c in df_scores_resampled.columns if c != "subject"]].groupby(df_scores_resampled.index).std()

# ---------- 1) VCVS Left: time-varying -> resample per subject, then average across subjects ----------
l_VCVS_left = []
for sub in df_features["subject"].unique():
    df_SC_L_sub, days_reg, cols_regions = get_img_SC_L(sub)  # df_reg has cols_regions + ['date']
    # index by timedelta from days_since_first
    df_SC_L_sub = df_SC_L_sub.drop(columns=["date"])
    df_SC_L_sub.index = pd.to_timedelta(days_reg, unit="D")
    # resample to FREQ
    df_SC_L_sub = df_SC_L_sub.resample(FREQ).mean()
    df_SC_L_sub = df_SC_L_sub[df_SC_L_sub.index < pd.Timedelta(days=600)]
    df_SC_L_sub["subject"] = sub
    l_VCVS_left.append(df_SC_L_sub)

df_VCVS_left = pd.concat(l_VCVS_left, ignore_index=False)
# subject-average at each resampled bin
df_VCVS_left_mean = (
    df_VCVS_left.drop(columns=["subject"])
    .groupby(df_VCVS_left.index)
    .mean()
)
df_VCVS_left_std = (
    df_VCVS_left.drop(columns=["subject"])
    .groupby(df_VCVS_left.index)
    .std()
)

# ---------- 2) DELTA CORRELATIONS: time-varying -> resample per subject, then average across subjects ----------
l_feat = []
for sub in df_features["subject"].unique():
    df_feat, days_feat, cols_features = get_img_delta_power(sub)  # df_feat only feature cols (scaled)
    df_feat.index = pd.to_timedelta(days_feat, unit="D")
    df_feat = df_feat.resample(FREQ).mean()
    df_feat = df_feat[df_feat.index < pd.Timedelta(days=600)]
    df_feat["subject"] = sub
    l_feat.append(df_feat)

df_delta_power_resampled = pd.concat(l_feat, ignore_index=False)
df_delta_power_resampled_mean = (
    df_delta_power_resampled.drop(columns=["subject"])
    .groupby(df_delta_power_resampled.index)
    .mean()
)

df_delta_power_resampled_std = (
    df_delta_power_resampled.drop(columns=["subject"])
    .groupby(df_delta_power_resampled.index)
    .std()
)
# reorder by regions order
cols_features = ['SC_L_delta', 'SC_R_delta', 'C_L_1_delta', 'C_L_2_delta', 'C_R_1_delta', 'C_R_2_delta']
df_delta_power_resampled_mean = df_delta_power_resampled_mean[cols_features]
df_delta_power_resampled_std = df_delta_power_resampled_std[cols_features]

# ---------- 3) YBOCS MAP: static per-subject 2D array -> simple subject-average ----------
maps_ybocs = []
for sub in df_features["subject"].unique():
    m = get_map_yboc_corr(sub)  # shape: (regions x features) or similar
    maps_ybocs.append(m)

map_ybocs_subject_mean = np.nanmean(np.stack(maps_ybocs, axis=0), axis=0)  # same shape as a single map
map_ybocs_subject_std = np.nanstd(np.stack(maps_ybocs, axis=0), axis=0)  # same shape as a single map

# ---------- 4) LABEL CORRELATIONS: static per-subject (labels x labels) -> subject-average ----------
maps_label_corr = []
for sub in df_features["subject"].unique():
    m = get_label_correlations(sub)  # shape: (n_labels x n_labels)
    maps_label_corr.append(m)

map_label_corr_subject_mean = np.nanmean(np.stack(maps_label_corr, axis=0), axis=0)
map_label_corr_subject_std = np.nanstd(np.stack(maps_label_corr, axis=0), axis=0)


plt.figure(figsize=(15, 15))

plt.rcParams.update({
    "font.size": 7,          # decrease font size (default ~12)
    "font.family": "Arial"    # set font to Arial
})

for sub_plt_idx, SUBJECT_PLT in enumerate(["aDBS009", "aDBS010"]):
    sub_offset_plt = sub_plt_idx * 5
    plt.subplot(4, 5, 1 + sub_offset_plt)
    img_scores = get_img_scores(SUBJECT_PLT)
    sns.heatmap(img_scores.T, annot=False, fmt=".0f", cmap='viridis', cbar_kws={"label": "Score values"}, vmin=0, vmax=100)
    plt.yticks(np.arange(len(cols_test))+0.5, cols_labels, rotation=0)

    plt.subplot(4, 5, 2 + sub_offset_plt)
    map_spec_region, days_since_first_map, cols_region = get_img_SC_L(SUBJECT_PLT)
    sns.heatmap(map_spec_region[cols_region].T, annot=False, fmt=".2f",
                cmap='viridis', cbar_kws={"label": "Feature values"}, vmin=-2, vmax=2)
    plt.yticks(np.arange(len(cols_region))+0.5, [c[len("SC_L_"):] for c in cols_region], rotation=0)
    plt.xticks(np.arange(len(days_since_first_map))+0.5, days_since_first_map, rotation=90)
    plt.title("VCVS Left")

    plt.subplot(4, 5, 3 + sub_offset_plt)
    map_spec_features, days_since_first_map_feature, cols_features = get_img_delta_power(SUBJECT_PLT)
    cols_features = ['SC_L_delta', 'SC_R_delta', 'C_L_1_delta', 'C_L_2_delta', 'C_R_1_delta', 'C_R_2_delta']

    sns.heatmap(map_spec_features[cols_features].T, annot=False, fmt=".2f", cmap='viridis',
                cbar_kws={"label": "Feature values"}, vmin=-2, vmax=2)
    plt.yticks(np.arange(len(cols_features))+0.5, [c[:-len("_delta")] for c in cols_features], rotation=0)
    plt.xticks(np.arange(len(days_since_first_map_feature))+0.5, days_since_first_map_feature, rotation=90)
    plt.title("Delta power")

    plt.subplot(4, 5, 4 + sub_offset_plt)
    map_neural_corr_ybocs = get_map_yboc_corr(SUBJECT_PLT)
    sns.heatmap(map_neural_corr_ybocs, annot=False, fmt=".2f", cmap='coolwarm', 
                xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef."},
    )
    plt.yticks(np.arange(len(regions))+0.5, regions, rotation=0)
    plt.xticks(np.arange(len(feature_names))+0.5, feature_names, rotation=90)

    plt.subplot(4, 5, 5 + sub_offset_plt)
    map_symptom_correlations = get_label_correlations(SUBJECT_PLT)
    sns.heatmap(map_symptom_correlations, annot=False, fmt=".2f", cmap='Reds', 
                xticklabels=cols_labels, yticklabels=cols_labels, cbar_kws={"label": "Pearson corr. coef."})
    plt.yticks(np.arange(len(cols_labels))+0.5, cols_labels, rotation=0)
    plt.xticks(np.arange(len(cols_labels))+0.5, cols_labels, rotation=90)

plt.subplot(4, 5, 1+ 10)
sns.heatmap(df_scores_resampled_mean.T, annot=False, fmt=".0f", cmap='viridis', cbar_kws={"label": "Score values"}, vmin=0, vmax=100)
plt.yticks(np.arange(len(cols_test))+0.5, cols_labels, rotation=0)
plt.xticks(np.arange(len(df_scores_resampled_mean.index))+0.5, df_scores_resampled_mean.index.days, rotation=90)
plt.title("MEAN")
plt.subplot(4, 5, 2 + 10)
sns.heatmap(df_VCVS_left_mean.T, annot=False, fmt=".2f", cmap='viridis', cbar_kws={"label": "Feature values"}, vmin=-2, vmax=2)
plt.yticks(np.arange(len(cols_region))+0.5, [c[len("SC_L_"):] for c in cols_region], rotation=0)
plt.xticks(np.arange(len(df_VCVS_left_mean.index))+0.5, df_VCVS_left_mean.index.days, rotation=90)
plt.title("MEAN")
plt.subplot(4, 5, 3 + 10)
sns.heatmap(df_delta_power_resampled_mean.T, annot=False, fmt=".2f", cmap='viridis', cbar_kws={"label": "Feature values"}, vmin=-2, vmax=2)
plt.yticks(np.arange(len(cols_features))+0.5, [c[:-len("_delta")] for c in cols_features], rotation=0)
plt.xticks(np.arange(len(df_delta_power_resampled_mean.index))+0.5, df_delta_power_resampled_mean.index.days, rotation=90)
plt.title("MEAN")
plt.subplot(4, 5, 4 + 10)
sns.heatmap(map_ybocs_subject_mean, annot=False, fmt=".2f", cmap='coolwarm', 
            xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef."},
            )
plt.yticks(np.arange(len(regions))+0.5, regions, rotation=0)
plt.xticks(np.arange(len(feature_names))+0.5, feature_names, rotation=90)
plt.title("MEAN")
plt.subplot(4, 5, 5 + 10)
sns.heatmap(map_label_corr_subject_mean, annot=False, fmt=".2f", cmap='Reds', 
            xticklabels=cols_labels, yticklabels=cols_labels, cbar_kws={"label": "Pearson corr. coef."})
plt.yticks(np.arange(len(cols_labels))+0.5, cols_labels, rotation=0)
plt.xticks(np.arange(len(cols_labels))+0.5, cols_labels, rotation=90)
plt.title("MEAN")

plt.subplot(4, 5, 1 + 15)
sns.heatmap(df_scores_resampled_std.T, annot=False, fmt=".2f", cmap='Reds', cbar_kws={"label": "Score values std"})
plt.yticks(np.arange(len(cols_test))+0.5, cols_labels, rotation=0)
plt.xticks(np.arange(len(df_scores_resampled_std.index))+0.5, df_scores_resampled_std.index.days, rotation=90)
plt.title("STD")
plt.subplot(4, 5, 2 + 15)
sns.heatmap(df_VCVS_left_std.T, annot=False, fmt=".2f", cmap='Reds', cbar_kws={"label": "Feature values std"})
plt.yticks(np.arange(len(cols_region))+0.5, cols_region, rotation=0)
plt.xticks(np.arange(len(df_VCVS_left_std.index))+0.5, df_VCVS_left_std.index.days, rotation=90)
plt.title("STD")
plt.subplot(4, 5, 3 + 15)
sns.heatmap(df_delta_power_resampled_std.T, annot=False, fmt=".2f", cmap='Reds', cbar_kws={"label": "Feature values std"})
plt.yticks(np.arange(len(cols_features))+0.5, cols_features, rotation=0)
plt.xticks(np.arange(len(df_delta_power_resampled_std.index))+0.5, df_delta_power_resampled_std.index.days, rotation=90)
plt.title("STD")
plt.subplot(4, 5, 4 + 15)
sns.heatmap(map_ybocs_subject_std, annot=False, fmt=".2f", cmap='Reds', 
            xticklabels=feature_names, yticklabels=regions, cbar_kws={"label": "Pearson corr. coef. std"},
            )
plt.yticks(np.arange(len(regions))+0.5, regions, rotation=0)
plt.xticks(np.arange(len(feature_names))+0.5, feature_names, rotation=90)
plt.title("YBOCS")
plt.subplot(4, 5, 5 + 15)
sns.heatmap(map_label_corr_subject_std, annot=False, fmt=".2f", cmap='Reds', 
            xticklabels=cols_labels, yticklabels=cols_labels, cbar_kws={"label": "Pearson corr. coef. std"})
plt.yticks(np.arange(len(cols_labels))+0.5, cols_labels, rotation=0)
plt.xticks(np.arange(len(cols_labels))+0.5, cols_labels, rotation=90)
plt.title("STD ")


