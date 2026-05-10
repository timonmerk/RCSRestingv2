import pandas as pd

df_scores = pd.read_csv("map_scores/YBOCS2_scores_aDBS012.csv")
df_scores["date"] = pd.to_datetime(df_scores["Date"])

# set x axis in days since first date make ticks every 500 days
df_scores["days_since_first"] = (df_scores["date"] - df_scores["date"].min()).dt.days

import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.plot(df_scores["days_since_first"], df_scores["YBOCS"], marker='o')
plt.title("YBOCS2 Scores Over Time for aDBS012")
plt.xlabel("Date")
plt.ylabel("YBOCS2 Score")
plt.xticks(rotation=90)
plt.ylim(0, 50)
plt.tight_layout()
plt.savefig("map_scores/YBOCS2_scores_aDBS012.pdf")
plt.show()