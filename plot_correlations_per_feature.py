import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
# i want create a multipage pdf with matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import permutation_test

df_corr = pd.read_csv("correlation_features.csv")
df_plt = df_corr.query("score_column == 'YBOCS II Total Score' and subject != 'ALL'") # channel.str.contains('VCVS')
df_plt["hem"] = df_plt["channel"].apply(lambda x: "left" if "left" in x else "right")
df_plt["loc"] = df_plt["channel"].apply(lambda x: "Cortex" if "Cortex" in x else "VCVS")
order_ = df_plt.query("loc == 'Cortex'").groupby(["feature_name"])["correlation"].mean().reset_index().sort_values("correlation", ascending=True)["feature_name"].tolist()
# get 5 lowest and 5 highest correlations
#order_plt = order_[:5] + order_[-5:]

order_plt = ["RawHjorth_Activity", "gamma", "burst_duration_delta_ms", "beta", "exponent", "delta", 
             "burst_amplitude_beta", "burst_amplitude_delta", "Sharpwave_Mean_interval_range_1_12", "offset"]

order_plt_sorted = sorted(order_plt, key=lambda x: order_.index(x))
from scipy.stats import permutation_test

hem_sel = "right"
df_plt = df_plt.query("hem == @hem_sel").reset_index(drop=True)

# Set up figure
plt.figure(figsize=(7, 7))
ax = plt.gca()

# Boxplot and swarmplot
sns.boxplot(
    data=df_plt,
    x="feature_name",
    y="correlation",
    hue="loc",
    order=order_plt_sorted,
    palette="viridis",
    showfliers=False,
    showmeans=True,
    boxprops=dict(alpha=0.5),
    ax=ax
)

sns.swarmplot(
    data=df_plt,
    x="feature_name",
    y="correlation",
    hue="loc",
    order=order_plt_sorted,
    dodge=True,
    color="black",
    alpha=0.5,
    legend=False,
    ax=ax
)

# Get unique locations
locs = df_plt["loc"].unique()
n_locs = len(locs)

# Significance thresholds
def get_significance_symbol(pval):
    if pval < 0.0001:
        return "****"
    elif pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return "n.s."

# Add permutation test results
for i, feat in enumerate(order_plt_sorted):
    for j, loc in enumerate(locs):
        subset = df_plt.query("feature_name == @feat and loc == @loc")["correlation"].dropna()
        if len(subset) < 2:
            continue

        # Permutation test against 0
        res = permutation_test(
            data=(subset, np.zeros_like(subset)),
            statistic=lambda x, y: np.mean(x - y),
            n_resamples=5000,
            alternative="two-sided",
            random_state=42
        )
        pval = res.pvalue
        symbol = get_significance_symbol(pval)

        # Calculate annotation position
        x = i + (j - 1) * 0.2  # x-position: left/right dodge for hue
        if j == 1:
            x += 0.25
        y = subset.max() + 0.05  # y-position just above the max value

        ax.text(x, y, symbol, ha="center", va="bottom", fontsize=10, color="black")

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"figures/correlation_features_modalities_unsigned_best_{hem_sel}.pdf")
plt.show()



plt.figure()
sns.boxplot(df_plt, x="feature_name", y="correlation", showfliers=False, order=order_plt_sorted, palette="viridis", hue="loc")
sns.swarmplot(df_plt, x="feature_name", y="correlation", order=order_plt_sorted, color="black", alpha=0.5, dodge=True, legend=False, hue="loc")
plt.tight_layout()
plt.xticks(rotation=90)


df_corr_best = df_corr.groupby(["subject", "score_column", "channel"])["correlation"].max().reset_index()
df_corr_best = (
    df_corr
      .groupby(["subject", "score_column", "channel"])          # group rows
      ["correlation"]                                           # pick the series
      .agg(lambda s: np.abs(s).max())                           # |·| → max
).reset_index()
df_corr_best["loc"] = ["Cortex" if "Cortex" in ch else "VCVS" for ch in df_corr_best["channel"]]
df_corr_best["hem"] = ["left" if "left" in ch else "right" for ch in df_corr_best["channel"]]
# get all subjects that have channel with Cortex_
subs_with_ecog = df_corr_best.query("channel.str.contains('Cortex_')", engine='python')["subject"].unique().tolist()
subs_with_ecog.remove("ALL")
df_corr_best = df_corr_best.query("subject in @subs_with_ecog").reset_index(drop=True)

cols_all = df_corr_best["score_column"].unique().tolist()

cols_of_interest = ["YBOCS II Total Score", "YBOCS II-Obsessions Sub-score", "YBOCS II-Compulsions Sub-score", "BDI-Total Score", 
                    "YMRS Total Score", 'Category 1:Concerns about Germs and Cotamination- Subscale Score',
                    'Category 2: Concerns about being Responsible for Harm, Injury, or Bad Luck- Subscale Scale',
                    'Category 3: Unacceptable Thoughts-Subscale Score',
                    'Category 4: Concerns about Symmetry, Completeness',
                    'Category 5: Sexually Intrusive Thoughts- Subscale Score',
                    'Category 6: Intrusive Violent Thoughts- Subscale Score',
                    'Category 7: Immoral and Scrupulous Thoughts- Subscale Score'
                    ]

#####################################################################################

order_ = ["VCVS", "Cortex"]
df_corr_best_ = df_corr_best.groupby(["loc", "subject", "score_column", "hem"])["correlation"].mean().reset_index()

cols_plt = ["YBOCS II Total Score", "YBOCS II-Obsessions Sub-score", "YBOCS II-Compulsions Sub-score", "BDI-Total Score", 
            'HDRS Total Score',
                "YMRS Total Score",]
df_plt = df_corr_best_.query("score_column in @cols_plt").reset_index(drop=True)

