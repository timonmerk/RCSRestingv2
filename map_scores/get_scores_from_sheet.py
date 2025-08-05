import pandas as pd
import numpy as np
import os
from datetime import datetime

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
    return dt_

def get_ybocs2(df_scores, date_check: np.datetime64, sub: str = "aDBS010", include_3d_jitter: bool = True):
    df = df_scores[sub]

    dates = df.iloc[4, :].values
    dates = [
        np.datetime64(f, 'D') if isinstance(f, datetime) else None
        for f in dates
    ]

    SCORE_ROWS = {
        "YBOCS II-Obsessions Sub-score": 13,
        "YBOCS II-Compulsions Sub-score": 14,
        "YBOCS II Insight Sub-Score ": 15,
        "YBOCS II Reliability Sub-Score": 16,
        "YBOCS II Global Severity Sub-Score": 17,
        "YBOCS II Global Improvement Sub-Score": 18,
        "YBOCS II Total Score": 19,
        "YBOCS I Obsessions Sub-Score": 20,
        "YBOCS I Compulsions Sub-Score": 21,
        "YBOCS I Insight Sub-Score": 22,
        "YBOCS I Total Score": 23,
        "HDRS Total Score": 24,
        "YMRS Total Score": 25,
        "Clinical Global Impression-Severity": 26,
        "Clinical Global Impression-Improvement": 27,
        "BDI-Total Score": 28,
        "BAI-Total Score": 29,
        "Composed-Anxious Sub Score": 31,
        "Agreeable-Hostile Sub Score": 32,
        "Elated-Depressed Sub Score": 33,
        "Confident-Unsure Sub Score": 34,
        "Energetic-Tired Sub Score": 35,
        "Clearheaded-Confused Sub Score": 36,
        "POMS Total Score": 37,
        "Category 1:Concerns about Germs and Cotamination- Subscale Score": 39,
        "Category 2: Concerns about being Responsible for Harm, Injury, or Bad Luck- Subscale Scale": 40,
        "Category 3: Unacceptable Thoughts-Subscale Score": 41,
        "Category 4: Concerns about Symmetry, Completeness": 42,
        "Category 5: Sexually Intrusive Thoughts- Subscale Score": 43,
        "Category 6: Intrusive Violent Thoughts- Subscale Score": 44,
        "Category 7: Immoral and Scrupulous Thoughts- Subscale Score": 45,
        "DOCS Total Score": 46,
        "SDS Total Score" : 47,
        "Attentional Impulsiveness-Subscale Score": 51,
        "Motor Impulsiveness- Subscale Score": 52,
        "Nonplanning Impulsiveness-Subscale Score": 53,
        "BIS Total Score": 54,
        "IUS Total Score": 55

    }

    date_check_with_time = np.datetime64(date_check, 'D')
    date_check_with_time = np.array(date_check_with_time, dtype='datetime64[D]')
    dates = np.array(dates, dtype='datetime64[D]')

    # check if sub == aDBS004 and date is 5/28/2020
    if sub == "aDBS004" and date_check_with_time == np.datetime64("2020-05-28"):
        # this date is not in the sheet, so we return None
        print("here")

    jitter = np.timedelta64(3, 'D')
    if include_3d_jitter:
        idx_ = np.where(
                np.logical_and((date_check_with_time >= dates - jitter),
                               (date_check_with_time <= dates + jitter))
        )[0]
    else:
        idx_ = np.where(dates == date_check_with_time)[0]
    if len(idx_) == 0:
        return None, None
    date__ = None

    rows_ = []
    if idx_.shape[0] > 1:
        for idx in idx_:
            date__ = dates[int(idx)]
            # return a pd.Series with the scores
            df_scores_d = []
            for key, row in SCORE_ROWS.items():
                df_scores_d.append(df.iloc[row, int(idx)])
            df_scores_d = list(pd.Series(df_scores_d, index=SCORE_ROWS.keys()))
            rows_.append(df_scores_d)
        df_rows = pd.DataFrame(rows_, dtype=object, columns=SCORE_ROWS.keys())
        # select the first value for each row if it's non nan, if nan, select the next one
        df_scores_d = df_rows.apply(lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan, axis=0)

    else:
        date__ = dates[int(idx_[0])]
        # return a pd.Series with the scores
        df_scores_d = []
        for key, row in SCORE_ROWS.items():
            df_scores_d.append(df.iloc[row, int(idx_[0])])
        df_scores_d = pd.Series(df_scores_d, index=SCORE_ROWS.keys())
    return date__, df_scores_d


PATH_DATA = "/scratch/timonmerk/restingstate_data"
list_files_csv = [os.path.join(PATH_DATA, f) for f in os.listdir(PATH_DATA) if f.endswith('.csv')]
subjects = [f[f.find("aDBS"): f.find("aDBS") + 7] for f in list_files_csv]
dates = [get_date(f) for f in list_files_csv]
dates_ = [convert_to_datetime(d) for d in dates]

df_scores  = {str(sub): pd.read_excel("aDBS Clinical Outcomes Master Database (all subjects) 10.30.2024.xlsx",
            sheet_name=sub, engine='openpyxl') for sub in np.unique(subjects)}

dates_, series_all_dfs = zip(*[
    get_ybocs2(df_scores, date_check=d, sub=subjects[i])
    for i, d in enumerate(dates_)
])

dates_ = np.array(dates_)
df_scores = pd.DataFrame(series_all_dfs)
df_scores["subject"] = subjects
df_scores["date"] = dates_
df_scores["file"] = list_files_csv

df_scores.to_csv("map_scores/scores_date_mapped.csv", index=False)

df_scores.query("subject == 'aDBS004'").groupby("date")["YBOCS II Total Score"].mean()

# sort by date
df_scores.query("subject == 'aDBS004'").sort_values(by="date")


files_aDBS004 = [f for f in list_files_csv if "aDBS004" in f]
dates_aDBS004 = [convert_to_datetime(get_date(f)) for f in files_aDBS004]
dates_aDBS004 = np.sort(dates_aDBS004)