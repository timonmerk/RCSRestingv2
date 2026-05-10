import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed


def get_date(f, ):
    idx_start = f.find("resting-state")
    idx_start = idx_start + f[idx_start:].find("_") + 1
    idx_sync_or_toggle = f.find("sync") if "sync" in f else f.find("toggle")
    if "BSoff" in f or "BSon" in f:
        idx_start = f.find("BSoff") + len("BSoff") + 1 if "BSoff" in f else f.find("BSon") + len("BSon") + 1
    return f[idx_start : idx_sync_or_toggle-1]

def convert_to_datetime(d):
    if d.startswith("20"):
        dt_ = datetime.strptime(d, "%Y%m%d%H%M%S%f")
    else:
        dt_ = datetime.fromtimestamp(int(d) / 1000)
    # keep only Ymd
    dt_ = dt_.strftime("%Y-%m-%d")
    return dt_

df_annot = pd.read_excel("annotations_rcs.xlsx", sheet_name="Sheet1")
duplicates = df_annot["Duplicate"].dropna().unique()
all_wrong = df_annot.query("Index == 'ALL'")["Filename"].unique()

df_scores = pd.read_csv("map_scores/scores_date_mapped.csv")
df_scores["file_short"] = [os.path.basename(f)[:-4] for f in df_scores["file"] ]
df_scores["file_short"] = df_scores["file_short"].str.replace(r'_stim_(left|right|NA)$', '', regex=True)
df_scores["file_short"] = df_scores["file_short"].str.replace(r'_(left|right)$', '', regex=True)

#SCORE_METRIC = 'YBOCS II Total Score'  # 'YBOCS II-Obsessions Sub-score', 'YBOCS II-Compulsions Sub-score'
SCORE_COLUMNS = list(df_scores.columns[:-4])
mapping_loc = {
    "C_0_left" : "Cortex",
    "C_1_left" : "Cortex",
    "C_0_right": "Cortex",
    "C_1_right": "Cortex",
    "SC_0_left": "VCVS",
    "SC_1_left": "VCVS",
    "SC_0_right": "VCVS",
    "SC_1_right": "VCVS",
    "SC_left_C_left": "Both",
    "SC_right_C_right": "Both",
    "SC_left_C_right": "Both",
    "SC_right_C_left": "Both",
    "C_left_right" : "Cortex",
    "SC_left_right": "VCVS"
}

new_ch_mapping = {
    "C_0_left" : "C_L_1",
    "C_1_left" : "C_L_2",
    "C_0_right": "C_R_1",
    "C_1_right": "C_R_2",
    "SC_0_left": "SC_L",
    "SC_1_left": "SC_L",
    "SC_0_right": "SC_R",
    "SC_1_right": "SC_R",
    "SC_left_C_left": "Misc",
    "SC_right_C_right": "Misc",
    "SC_left_C_right": "Misc",
    "SC_right_C_left": "Misc",
    "C_left_right" : "Misc",
    "SC_left_right": "Misc"
}

READ_FEATURES_COMBINED = False

if READ_FEATURES_COMBINED:
    PATH_READ = "features_out"
    files = [f for f in os.listdir(PATH_READ) if "features_prep.csv" in f]
    dfs_ = []
    for file in tqdm(files):
        f_name_find = file[:file.find("_features_prep")]
        if f_name_find in duplicates or f_name_find in all_wrong:
            continue
        df = pd.read_csv(os.path.join(PATH_READ, file))
        df_annot_file = df_annot.query("Filename == @f_name_find")

        df_filtered = df[~df["idx"].isin(df_annot_file["Index"].values)]
        if df_filtered.empty:
            print(f"File {file} is empty after filtering. Skipping.")
            continue
        df_g = df_filtered.groupby(["subject", "channel", "feature_name"])["feature_value"].median().reset_index()
        df_g["loc"] = df_g["channel"].apply(
            lambda x: mapping_loc[x] if x in mapping_loc else "Unknown"
        )
            
        df_g["hemisphere"] = df_g["channel"].apply(
            lambda x: "Left" if "left" in x and "right" not in x else
                    "Right" if "right" in x and "left" not in x else
                    "Both"
        )

        df_g["new_ch"] = df_g["channel"].apply(
            lambda x: new_ch_mapping[x] if x in new_ch_mapping else "Unknown"
        )

        df_g["file"] = f_name_find
        subject = f_name_find.split("_")[0]
        # replace combined_timeseries or combined_timeseries_lr with nothing
        f_name_find_name = f_name_find.replace("_combined_timeseries", "").replace("_combined_lr_timeseries", "")
        #df_g["score"] = df_filtered["score"].unique()[0]
        scores_series = pd.Series(
            df_scores.query("subject == @subject and file_short == @f_name_find_name")[SCORE_COLUMNS].values[0], 
            index=SCORE_COLUMNS)
        df_g["subject"] = subject
        # add scores_series to df_g
        for col in SCORE_COLUMNS:
            df_g[col] = scores_series[col]

        #df_g["score"] = scores_series


        dfs_.append(df_g)
    df_ = pd.concat(dfs_, ignore_index=True)
    df_.to_csv("features_prep_combined.csv", index=False)
else:
    df_ = pd.read_csv("features_prep_combined.csv")

GROUP_FEATURES_FOR_DECODING = True

if GROUP_FEATURES_FOR_DECODING:
    def get_date(f, ):
        idx_start = f.find("resting-state")
        idx_start = idx_start + f[idx_start:].find("_") + 1
        idx_sync_or_toggle = f.find("sync") if "sync" in f else f.find("toggle")
        if "BSoff" in f or "BSon" in f:
            idx_start = f.find("BSoff") + len("BSoff") + 1 if "BSoff" in f else f.find("BSon") + len("BSon") + 1
        return f[idx_start : idx_sync_or_toggle-1]

    def convert_to_datetime(d):
        if d.startswith("20"):
            dt_ = datetime.strptime(d, "%Y%m%d%H%M%S%f")
        else:
            dt_ = datetime.fromtimestamp(int(d) / 1000)
        # keep only Ymd
        dt_ = dt_.strftime("%Y-%m-%d")
        return dt_


    # --- Load data ---
    df_ = pd.read_csv("features_prep_combined.csv")

    # --- 1. Ensure SCORE_COLUMNS are numeric in one shot ---
    df_[SCORE_COLUMNS] = df_[SCORE_COLUMNS].apply(pd.to_numeric, errors='coerce')

    # --- 2. Parse info from "file" column once ---
    # assumes "file" looks like "<subject>_<rs_name>_..."
    file_parts = df_["file"].str.split("_", n=2, expand=True)
    df_["subject"] = file_parts[0]
    df_["rs_name"] = file_parts[1]

    # date: keep your functions but call them only once on df_
    df_["date"] = df_["file"].apply(lambda x: convert_to_datetime(get_date(x)))

    # --- 3. Aggregate per file / feature / new_ch ---
    group_cols = ["file", "feature_name", "new_ch"]

    # dictionary for aggregations: mean for scores + feature_value, first for channel
    agg_dict = {c: "mean" for c in SCORE_COLUMNS + ["feature_value"]}
    agg_dict["channel"] = "first"

    df_features = (
        df_
        .groupby(group_cols, as_index=False)
        .agg(agg_dict)
    )

    # --- 4. Pivot to wide format on feature_name ---
    df_features_wide = (
        df_features
        .pivot_table(
            index=["file", "channel"],
            columns="feature_name",
            values="feature_value"
        )
        .reset_index()
    )

    # after pivot, "feature_name" is in the column index level, so flatten if needed
    df_features_wide.columns.name = None  # remove the pivot column name

    # --- 5. Attach metadata (date, subject, rs_name) per file ---
    df_meta = (
        df_[["file", "subject", "rs_name", "date"]]
        .drop_duplicates("file")
    )

    df_features_wide = df_features_wide.merge(df_meta, on="file", how="left")

    # --- 6. Map new_ch from channel ---
    df_features_wide["new_ch"] = df_features_wide["channel"].map(
        lambda x: new_ch_mapping.get(x, "Unknown")
    )

    # --- 7. Attach SCORE_COLUMNS (one row per file) without iterrows ---
    # if multiple rows per file, take the first (matches your previous `iloc[0]`)
    df_scores = (
        df_[["file"] + SCORE_COLUMNS]
        .groupby("file", as_index=False)
        .first()
    )

    df_save = df_features_wide.merge(df_scores, on="file", how="left")

    # --- 8. Save ---
    df_save.to_csv("features_prep_wide_per_rs_sess.csv", index=False)

    ## IGNORE FROM HERE ON
    # couple of debugging steps:
    # df_features.query("subject == 'aDBS004' and date == '2020-02-18'")[["channel", "rs_name"]]
    # df_features.query("channel == 'SC_0_left'").groupby(["subject", "date"])["rs_name"].count().reset_index().iloc[50:100]


    # cols_features = [c for c in df_save.columns if c not in ["file", "date", "subject", "new_ch", "channel", "rs_name"] + SCORE_COLUMNS]
    # #df_features_sess = df_features.copy()
    # #df_features_sess.to_csv("df_sess_split_by_rs.csv")
    # df_features = df_save.groupby(["subject", "date", "new_ch", "rs_name"])[cols_features].mean().reset_index()

    #     #col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
    # for col_score in SCORE_COLUMNS:
    #     df_features[col_score] = df_.groupby(["file", "new_ch"])[col_score].mean().values

    # df_features = df_features.reset_index(drop=True)
    # df_features.to_csv("features_prep_wide_per_rs_sess.csv")

    # df_features = df_features.groupby(["subject", "date", "new_ch"])[cols_features].mean().reset_index()
    # for col_score in SCORE_COLUMNS:
    #     df_features[col_score] = df_.groupby(["subject", "date", "new_ch"])[col_score].mean().values

    # # Step 2: Rename columns to include 'new_ch_' prefix
    # df_features_renamed = df_features.copy()
    # for col in cols_features:
    #     df_features_renamed[col] = df_features_renamed[col]
    #     df_features_renamed.rename(columns={col: f"{col}_{{new_ch}}" }, inplace=True)  # We'll handle this in pivot

    # # Step 3: Pivot table to wide format
    # df_pivot = df_features.pivot(index=["subject", "date"], columns="new_ch", values=cols_features)

    # # Step 4: Flatten MultiIndex columns
    # df_pivot.columns = [f"{ch}_{feat}" for feat, ch in df_pivot.columns]

    # # Step 5: Reset index if needed
    # df_pivot = df_pivot.reset_index()
    # for col_score in SCORE_COLUMNS:
    #     df_pivot[col_score] = df_features.groupby(["subject", "date"])[col_score].mean().values
    # df_pivot.to_csv("features_prep_combined_wide.csv", index=False)

features_names = df_["feature_name"].unique()
chs_plt = ["VCVS_left", "VCVS_right", "Cortex_left", "Cortex_right"]
chs_plt = df_["new_ch"].unique().tolist()
#plt.figure()

def process_feature(score, ch_plt, feature_, df_):
    results = []
    try:
        for sub in df_["subject"].unique():
            df_sub = df_.query("subject == @sub and feature_name == @feature_ and new_ch == @ch_plt").copy()
            df_sub["date"] = df_sub["file"].apply(lambda x: convert_to_datetime(get_date(x)))
            df_f_g_ = df_sub.groupby("date")[["feature_value", score]].mean()

            if df_f_g_.shape[0] < 2:
                continue

            idx_not_na = df_f_g_.index[df_f_g_[score].notna() & df_f_g_["feature_value"].notna()]
            if len(idx_not_na) < 2 or df_f_g_["feature_value"].nunique() == 1 or df_f_g_[score].nunique() == 1:
                continue

            df_f_g_ = df_f_g_.loc[idx_not_na]
            corr, p = stats.pearsonr(df_f_g_["feature_value"], df_f_g_[score])
            if np.isnan(corr) or np.isnan(p):
                continue
            results.append({
                "subject": sub,
                "feature_name": feature_,
                "channel": ch_plt,
                "correlation": corr,
                "p_value": p,
                "score_column": score
            })

        # All subjects
        df_all = df_.query("feature_name == @feature_ and new_ch == @ch_plt").copy()
        df_f_g_all = df_all.groupby("file")[["feature_value", score]].mean()
        idx_not_na = df_f_g_all.index[df_f_g_all[score].notna() & df_f_g_all["feature_value"].notna()]
        if len(idx_not_na) < 2 or df_f_g_all["feature_value"].nunique() == 1 or df_f_g_all[score].nunique() == 1:
            return results

        df_f_g_all = df_f_g_all.loc[idx_not_na]
        corr, p = stats.pearsonr(df_f_g_all["feature_value"], df_f_g_all[score])
        results.append({
            "subject": "ALL",
            "feature_name": feature_,
            "channel": ch_plt,
            "correlation": corr,
            "p_value": p,
            "score_column": score
        })
    except Exception as e:
        print(f"Error: {feature_} / {score} / {ch_plt} → {e}")
    return results

unimodal_features = [
    f for f in df_["feature_name"].unique()
    if not("_C_" in f and "_SC_" in f)
    and "_corr_" not in f
    and "fft_psd" not in f
]

tasks = [
    (score, ch_plt, feature_)
    for score in SCORE_COLUMNS
    for ch_plt in chs_plt
    for feature_ in unimodal_features
]

with tqdm_joblib(tqdm(desc="Computing correlations", total=len(tasks))):
    df_res_nested = Parallel(n_jobs=40)(
        delayed(process_feature)(score, ch_plt, feature_, df_)
        for score, ch_plt, feature_ in tasks
    )

from itertools import chain

df_res = list(chain.from_iterable(df_res_nested))
df_res = pd.DataFrame(df_res)

df_res.to_csv("correlation_features.csv", index=False)

# add a subject ALL, for which the correlation across all subjects is calculated


# show in four boxes
score_column = "YBOCS II-Compulsions Sub-score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
plt.figure(figsize=(12, 15))
for i, ch in enumerate(df_res["channel"].unique()):
    df_ch = df_res.query("channel == @ch and score_column == @score_column")
    feature_name_order = df_ch.groupby("feature_name")["correlation"].mean().sort_values().index.tolist()
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=df_ch.query("subject != 'ALL'"), x="feature_name", y="correlation", order=feature_name_order, showmeans=True, showfliers=False, boxprops=dict(facecolor='none', edgecolor='black'))
    sns.swarmplot(data=df_ch.query("subject != 'ALL'"), x="feature_name", y="correlation", color=".25", alpha=0.5, order=feature_name_order)
    sns.swarmplot(data=df_ch.query("subject == 'ALL'"), x="feature_name", y="correlation", color="red", alpha=1, order=feature_name_order)
    plt.title(ch)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"figures/correlation_features_ordered_{score_column}.pdf")




df_res_across = []
bimodal_features = [f for f in df_["feature_name"].unique() if "_corr_" in f]
for feature_ in bimodal_features:
    for sub in df_["subject"].unique():
        df_sub = df_.query("subject == @sub and feature_name == @feature_ and new_ch == 'Misc'")
        df_f_g_ = df_sub.groupby("file")[["feature_value", "score"]].mean()
        if df_f_g_.shape[0] < 2:
            continue
        corr, p = stats.pearsonr(df_f_g_["feature_value"], df_f_g_["score"])
        df_res_across.append({
            "subject": sub,
            "feature_name": feature_,
            "channel": ch_plt,
            "correlation": corr,
            "p_value": p
        })
    df_all = df_.query("feature_name == @feature_ and new_ch == 'Misc'")
    df_f_g_all = df_all.groupby("file")[["feature_value", "score"]].mean()
    corr, p = stats.pearsonr(df_f_g_all["feature_value"], df_f_g_all["score"])
    df_res_across.append({
        "subject": "ALL",
        "feature_name": feature_,
        "channel": ch_plt,
        "correlation": corr,
        "p_value": p
    })
df_res_across = pd.DataFrame(df_res_across)

plt.figure(figsize=(8, 6))
order_ = df_res_across.groupby("feature_name")["correlation"].mean().sort_values().index.tolist()
sns.boxplot(data=df_res_across.query("subject != 'ALL'"), x="feature_name", y="correlation", showmeans=True, showfliers=False, boxprops=dict(facecolor='none', edgecolor='black'), order=order_)
sns.swarmplot(data=df_res_across.query("subject != 'ALL'"), x="feature_name", y="correlation", color=".25", alpha=0.5, order=order_)
sns.swarmplot(data=df_res_across.query("subject == 'ALL'"), x="feature_name", y="correlation", color="red", alpha=1, order=order_)
plt.title("Bimodal Features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figures/correlation_features_bimodal_ordered.pdf")

plt.figure()
df_scores_plt = df_.groupby(["subject", "file"])["score"].mean().reset_index()
sns.boxplot(data=df_scores_plt, x="subject", y="score", showmeans=True, showfliers=False)
sns.swarmplot(data=df_scores_plt, x="subject", y="score", color=".25", alpha=0.5)
plt.title("YBOCS-II per Subject")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figures/scores_per_subject.pdf")

## add a barplot with the sum of scores per subject
df_scores_sum = df_scores_plt.groupby("subject")["score"].count().reset_index()
plt.figure(figsize=(6, 4))
sns.barplot(data=df_scores_sum, x="subject", y="score", palette="viridis")
plt.title("Counts of YBOCS-II Scores per Subject")
plt.xticks(rotation=90)
plt.xlabel("Subject")
plt.ylabel("Count of #Scores")
plt.tight_layout()
plt.savefig("figures/scores_sum_per_subject.pdf")
