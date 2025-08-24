import pandas as pd
import os

PATH_ = "/Users/Timon/Library/CloudStorage/Box-Box/APL_BCM_Share_SUDS/Resting-State-Videos"

subs = [f for f in os.listdir(PATH_) if os.path.isdir(os.path.join(PATH_, f))]

l_ = []
for sub in subs:
    path_sub = os.path.join(PATH_, sub)
    dates = [f for f in os.listdir(path_sub) if os.path.isdir(os.path.join(path_sub, f))]
    for date in dates:
        path_vid = os.listdir(os.path.join(path_sub, date))[0]
        l_.append({
            "sub" : sub,
            "date" : date,
            "video" : path_vid,
        })

df_ = pd.DataFrame(l_)
df_.to_csv("FAUS_rs/fau_date_video_mapping.csv", index=False)

