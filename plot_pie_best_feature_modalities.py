import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
# i want create a multipage pdf with matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import permutation_test


df_corr = pd.read_csv("correlation_features.csv")
df_corr_best = df_corr.groupby(["subject", "score_column", "channel"])["correlation"].max().reset_index()
df_corr_best = (
    df_corr
      .groupby(["subject", "score_column", "channel"])          # group rows
      ["correlation"]                                           # pick the series
      .agg(lambda s: np.abs(s).max())                           # |·| → max
).reset_index()

idx_max = df_corr.groupby(["subject", "score_column", "channel"])["correlation"]\
                 .apply(lambda s: s.abs().idxmax())

df_corr_best = df_corr.loc[idx_max].reset_index(drop=True)
df_corr_best["correlation"] = df_corr_best["correlation"].abs()

df_corr_best["loc"] = ["Cortex" if "Cortex" in ch else "VCVS" for ch in df_corr_best["channel"]]
df_corr_best["hem"] = ["left" if "left" in ch else "right" for ch in df_corr_best["channel"]]
# get all subjects that have channel with Cortex_
subs_with_ecog = df_corr_best.query("channel.str.contains('Cortex_')", engine='python')["subject"].unique().tolist()
subs_with_ecog.remove("ALL")
df_corr_best = df_corr_best.query("subject in @subs_with_ecog").reset_index(drop=True)
cols_plt = ["YBOCS II Total Score", "YBOCS II-Obsessions Sub-score", "YBOCS II-Compulsions Sub-score", "BDI-Total Score", 
            'HDRS Total Score',
                "YMRS Total Score",]
df_corr_best = df_corr_best.query("score_column in @cols_plt").reset_index(drop=True)

cols_all = df_corr_best["score_column"].unique().tolist()

plt.figure(figsize=(10, 6))
for loc_idx, loc in enumerate(["Cortex", "VCVS"]):
    plt.subplot(1, 2, loc_idx + 1)
    counts = df_corr_best.query(f"loc == '{loc}'")["feature_name"].value_counts()
    total = counts.sum()
    percentages = (counts / total) * 100

    # Separate major and minor features
    major = percentages[percentages >= 5]
    minor = percentages[percentages < 5]

    # Prepare final percentages with "Others" last
    combined = major.sort_values(ascending=False)
    if not minor.empty:
        combined["Others"] = minor.sum()

    # Match back to absolute counts
    final_counts = combined * total / 100

    # Generate reversed viridis colormap (brightest for highest %)
    colors = sns.color_palette("viridis", len(combined))
    colors = colors[::-1]

    plt.pie(final_counts, labels=combined.index, autopct='%1.1f%%',
            startangle=90, counterclock=False, colors=colors)
    plt.title(f"Feature Name Distribution in {loc} (≥5% shown)")
    plt.axis('equal')
plt.tight_layout()
plt.savefig("figures/ECoG_SubCortex_feature_name_distribution.pdf")
plt.show()
