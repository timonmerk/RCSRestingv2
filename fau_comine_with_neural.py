import pandas as pd
from tqdm import tqdm

df_features = pd.read_csv("features_prep_combined_wide.csv")

df_mapping = pd.read_csv("FAUS_rs/fau_date_video_mapping.csv")
subs = ["004", "005", "007", "008", "009", "010", "011", "012"]

rows_ = []
for sub in tqdm(subs):
    sub = "008"
    df_FAU_sub = pd.read_csv(f"FAUS_rs/{sub}_FAU.csv")
    AU_cols = [col for col in df_FAU_sub.columns if col.startswith("AU")]
    df_FAU_sub_mean = df_FAU_sub.query("face_detected == 1").groupby("identifier")[AU_cols].mean().reset_index()
    if sub == "008":
        df_FAU_sub_sess_ = pd.read_csv(f"FAUS_rs/{sub}_rerun_FAU.csv")
        df_FAU_sub_mean_sess_ = df_FAU_sub_sess_.query("face_detected == 1").groupby("identifier")[AU_cols].mean().reset_index()
    
    sub_int = int(sub)
    df_sub = df_mapping.query("sub == @sub_int")
    
    for row in df_FAU_sub_mean.itertuples():
        identifier = row.identifier
        video_name = identifier.split("_")[1]+".MP4"
        if sub == "008" and video_name == "GH011116.MP4":
            continue

        date = df_sub.query("video == @video_name")["date"].values
        if len(date) == 0:
            continue
        date = date[0]

        sub_adbs_name = f"aDBS{sub}"
        row_use = df_features.query("subject == @sub_adbs_name and date == @date")
        if row_use.empty:
            continue
    
        row_dict = row_use.iloc[0].to_dict()
        row_dict.update({f"FAU_{col}": getattr(row, col) for col in AU_cols})
        rows_.append(row_dict)

    # there is no date anyways that matches for the two 008 dates; so it wasn't included anyways...
    if sub == "008":
        for row in df_FAU_sub_mean_sess_.itertuples():
            identifier = row.identifier
            video_name = identifier.split("_")[1]+".MP4"

            if identifier == "cropped_GH011116_11_08":
                date = "2021-11-08"
            else:
                date = "2021-11-12"

            sub_adbs_name = f"aDBS{sub}"
            row_use = df_features.query("subject == @sub_adbs_name and date == @date")
            if row_use.empty:
                continue
        
            row_dict = row_use.iloc[0].to_dict()
            row_dict.update({f"FAU_{col}": getattr(row, col) for col in AU_cols})
            rows_.append(row_dict)

df_fau = pd.DataFrame(rows_)
df_fau.to_csv("FAUS_rs/fau_neural_combined.csv", index=False)