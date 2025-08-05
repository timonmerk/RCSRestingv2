from datetime import datetime
import pandas as pd
from functools import reduce

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

def preprocess_features(df, col_score):
    df__ = []

    for ch_loc in ["VCVS_left", "VCVS_right", "Cortex_left", "Cortex_right", "Misc"]:
        df_loc = df.query("new_ch == @ch_loc")
        df_features = df_loc.groupby(["file", "feature_name"])[[col_score, "feature_value"]].mean().reset_index()
        df_features["date"] = df_features["file"].apply(get_date)
        df_features["date"] = df_features["date"].apply(convert_to_datetime)
        df_features["subject"] = df_features["file"].apply(lambda x: x.split("_")[0])

        df_features_ = df_features.pivot_table(index=["date"], columns="feature_name", values="feature_value").reset_index()
        df_features_[col_score] = df_features.groupby("date")[col_score].mean().values
        file_by_date = df_features.groupby("date")["file"].first()
        df_features_["file"] = df_features_["date"].map(file_by_date)
        df_features_["subject"] = df_features_["file"].apply(lambda x: x.split("_")[0])
        df_features_ = df_features_.reset_index(drop=True)
        df_features_["loc"] = ch_loc
        cols_features = [f"{ch_loc}_{c}" for c in df_features_.columns if c not in ["file", "date", "subject", col_score, "loc"]]
        feature_cols = [c for c in df_features_.columns if c not in ["file", "date", "subject", col_score, "loc"]]

        # Build a renaming dictionary: old_name -> new_name
        rename_dict = {col: f"{ch_loc}_{col}" for col in feature_cols}

        # Rename the columns
        df_features_.rename(columns=rename_dict, inplace=True)
        if ch_loc != "VCVS_left":
            # remove loc and score_column from the columns
            df_features_.drop(columns=["loc", col_score, "file"], inplace=True)

        df__.append(df_features_)

    df_all = reduce(lambda left, right: pd.merge(
        left, right, on=["subject", "date"], how="outer"), df__)


    score_thresholds = {
        "aDBS004" : 49 * 0.65,
        "aDBS005" : 46 * 0.65,
        "aDBS007" : 47 * 0.65,
        "aDBS008" : 37 * 0.65,
        "aDBS009" : 45 * 0.65,
        "aDBS010" : 37 * 0.65,
        "aDBS011" : 45 * 0.65,
        "aDBS012" : 46 * 0.65,
    }

    # add a new column responde that is 1 if the score is above the threshold, else 0
    df_all["response"] = 0
    for sub, threshold in score_thresholds.items():
        df_all.loc[df_all["subject"] == sub, "response"] = (df_all.query("subject == @sub")["YBOCS II Total Score"] > threshold).astype(int)

    return df_all