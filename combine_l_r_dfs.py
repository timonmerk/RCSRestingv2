PATHS_ = "/Volumes/labworlds/Provenza/MSIT_analysis/Raw_Data/aDBS008/2021-12-03/lfp_left_data.csv"

import pandas as pd
from matplotlib import pyplot as plt


df_left = pd.read_csv(PATHS_)
df_right = pd.read_csv("/Volumes/labworlds/Provenza/MSIT_analysis/Raw_Data/aDBS008/2021-12-03/lfp_right_data.csv")

# df_left["localTime"] = pd.to_datetime(df_left["localTime"])
# df_left["localTime"].diff().plot()
# plt.show()


df_left["localTime"] = pd.to_datetime(df_left["localTime"])
df_right["localTime"] = pd.to_datetime(df_right["localTime"])

# merge on localTime
df_merged = pd.merge_asof(df_left.sort_values("localTime"), df_right.sort_values("localTime"), on="localTime", direction="nearest", tolerance=pd.Timedelta("1ms"), suffixes=("_left", "_right"))