import os
import numpy as np
import pandas as pd

PATH_ = "/scratch/timonmerk/rs_prep"
files_ = [f for f in os.listdir(PATH_) if f.endswith(".csv")]
files_map = [f[:-len("combined_lr.csv")] if "combined_lr.csv" in f else f[:-len("combined.csv")] for f in files_]


scores = pd.read_csv("map_scores/scores_date_mapped.csv")
scores["fname_trim"] = scores.file.apply(lambda x: os.path.basename(x))

def cut_f(f):
    if "stim_NA.csv" in f:
        return f[:-len("stim_NA.csv")]
    elif "stim_right.csv" in f:
        return f[:-len("stim_right.csv")]
    elif "stim_left.csv" in f:
        return f[:-len("stim_left.csv")]
    elif "_left.csv" in f:
        return f[:-len("left.csv")]
    elif "_right.csv" in f:
        return f[:-len("right.csv")]
    return f

scores["fname_trim"] = scores.fname_trim.apply(cut_f)
cols_scores = list(scores.columns[:-4])

scores_ = []
for f in files_map:
    if f in scores.fname_trim.values:
        idx = scores.fname_trim.values.tolist().index(f)
        scores_.append(scores.iloc[idx][cols_scores])
    else:
        scores_.append(None)

df_files = pd.DataFrame(scores_)
df_files["fname_trim"] = files_map
df_files["file"] = files_

# df_files = pd.DataFrame({
#     "file": files_,
#     "ybocs": scores_,
#     "fname_trim": files_map
# })
# check were ybocs is not np.nan
#df_files = df_files[df_files.ybocs.notna()]

df_files.to_csv("map_scores/files_ybocs.csv", index=False)
df_files["sub"] = df_files.file.apply(lambda x: x.split("_")[0])

import seaborn as sns
from matplotlib import pyplot as plt
sns.boxplot(data=df_files, x="sub", y="ybocs", order=np.sort(df_files["sub"].unique()), showfliers=False)
sns.swarmplot(data=df_files, x="sub", y="ybocs", color=".25", alpha=0.5, order=np.sort(df_files["sub"].unique()))
plt.savefig("boxplot_ybocs.pdf", bbox_inches='tight')
plt.close()
