import pandas as pd

# ok, don't use this scale, but the original one

PATH_RS = "/Users/Timon/Documents/Houston/whisper/audio_neural_features_combined_rs.csv"
df_rs = pd.read_csv(PATH_RS)

col_ybocs = 'YBOCS II Total Score'
df_rs["date"] = pd.to_datetime(df_rs["date"])
# compute days since first date for each "subject" 
df_rs["days_since_first"] = (df_rs["date"] - df_rs.groupby("subject")["date"].transform("min")).dt.days

# run correlation between ybocs and days since first date for each subject
correlations = df_rs.groupby("subject").apply(lambda x: x[col_ybocs].corr(x["days_since_first"]))
correlations.to_csv("correlations_ybocs_days_since_first.csv")
# make a quick time plot of YBOCS scores over time for each subject, show them all in different panels, group by time
import matplotlib.pyplot as plt
subjects = df_rs["subject"].unique()
n_subjects = len(subjects)
fig, axes = plt.subplots(n_subjects // 2 + n_subjects % 2, 2, figsize=(10, n_subjects * 2))
for i, subject in enumerate(subjects):
    ax = axes[i // 2, i % 2]
    df_subject = df_rs[df_rs["subject"] == subject]
    # sort by days since first date
    df_subject = df_subject.sort_values("days_since_first")
    ax.plot(df_subject["days_since_first"], df_subject[col_ybocs], marker='o')
    ax.set_title(f"Subject {subject} (corr={correlations[subject]:.2f})")
    ax.set_xlabel("Days Since First Date")
    ax.set_ylabel("YBOCS II Total Score")
    ax.set_ylim(0, 50)
plt.tight_layout()
plt.show()