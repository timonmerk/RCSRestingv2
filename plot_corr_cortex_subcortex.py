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

plt.figure(figsize=(7, 7))
ax = plt.gca()

# Base boxplot and swarmplot
# 
sns.boxplot(data=df_plt, x="score_column", y="correlation",
            hue="loc", order=cols_plt, hue_order=order_, showfliers=False, ax=ax, palette="viridis", boxprops=dict(alpha=0.5), showmeans=True)
# for patch in ax.artists:
#     patch.set_alpha(0.2)
sns.swarmplot(data=df_plt, x="score_column", y="correlation",
                hue="loc", order=cols_plt, hue_order=order_, dodge=True,
                palette=["gray", "gray"], legend=False)

for idx, score in enumerate(cols_plt):
    df_score = df_plt[df_plt["score_column"] == score]

    # Ensure both VCVS and Cortex are present somewhere
    if df_score["loc"].nunique() < 2:
        continue

    # Pivot on both subject and hemisphere
    df_pivot = df_score.pivot_table(index=["subject", "hem"], columns="loc", values="correlation")

    # Skip if too little data or if required columns are missing
    if df_pivot.shape[0] <= 1 or set(["VCVS", "Cortex"]) - set(df_pivot.columns):
        continue

    # Draw lines per (subject, hem) pair
    for (subj, hem), row in df_pivot.dropna().iterrows():
        # Slight offset per hemisphere to avoid full overlap
        offset = -0.2 #if hem == "left" else 0.1
        x_vals = [idx + offset, idx + offset + 0.4]
        y_vals = [row["VCVS"], row["Cortex"]]
        linestyle = '-' if hem == "left" else '--'
        ax.plot(x_vals, y_vals, color="black", linewidth=1, alpha=0.5, linestyle=linestyle)

    # Combine all valid VCVS–Cortex pairs for stat
    vcvs_vals = df_pivot["VCVS"].dropna()
    cortex_vals = df_pivot["Cortex"].dropna()

    if len(vcvs_vals) == len(cortex_vals):
        res = permutation_test(
            (vcvs_vals, cortex_vals),
            statistic=lambda x, y: np.mean(x - y),
            n_resamples=10000,
            alternative='two-sided',
            random_state=42,
        )
        pval = res.pvalue
        if pval < 0.0001:
            symbol = "****"
        elif pval < 0.001:
            symbol = "***"
        elif pval < 0.01:
            symbol = "**"
        elif pval < 0.05:
            symbol = "*"
        else:
            symbol = "n.s." 

        # Add single significance marker
        y_max = df_score["correlation"].max()
        ax.text(idx, y_max + 0.05, symbol,
                ha='center', va='bottom', fontsize=11, color='black',)

# despine the plot
sns.despine(trim=False, right=True, top=True)
plt.xticks(rotation=90)
plt.ylabel("Correlation with scores")
plt.title("Correlation of features with scores")
ax.legend(title="Channel", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.ylim(0.2, 1.05)
#plt.yticks(np.arange(0, 1.1, 0.1))
# limit xticks only to first 10 characters of current labels
#plt.xticks(np.arange(len(cols_plt)), [f[:10] for f in cols_plt]
plt.savefig("figures/ECoG_SubCortex_correlations_limited.pdf")


#####################################################################################
# ALL scores

pdf_ = PdfPages("figures/ECoG_SubCortex_correlations.pdf")
order_ = ["VCVS", "Cortex"]
df_corr_best_ = df_corr_best.groupby(["loc", "subject", "score_column", "hem"])["correlation"].mean().reset_index()
for i in np.arange(0, 35, 7):
    cols_plt = cols_all[i:i+7]
    df_plt = df_corr_best_.query("score_column in @cols_plt").reset_index(drop=True)

    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Base boxplot and swarmplot
    # 
    sns.boxplot(data=df_plt, x="score_column", y="correlation",
                hue="loc", order=cols_plt, hue_order=order_, showfliers=False, ax=ax, palette="viridis", boxprops=dict(alpha=0.5), showmeans=True)
    # for patch in ax.artists:
    #     patch.set_alpha(0.2)
    sns.swarmplot(data=df_plt, x="score_column", y="correlation",
                  hue="loc", order=cols_plt, hue_order=order_, dodge=True,
                  color=".25", legend=False, ax=ax)

    for idx, score in enumerate(cols_plt):
        df_score = df_plt[df_plt["score_column"] == score]

        # Ensure both VCVS and Cortex are present somewhere
        if df_score["loc"].nunique() < 2:
            continue

        # Pivot on both subject and hemisphere
        df_pivot = df_score.pivot_table(index=["subject", "hem"], columns="loc", values="correlation")

        # Skip if too little data or if required columns are missing
        if df_pivot.shape[0] <= 1 or set(["VCVS", "Cortex"]) - set(df_pivot.columns):
            continue

        # Draw lines per (subject, hem) pair
        for (subj, hem), row in df_pivot.dropna().iterrows():
            # Slight offset per hemisphere to avoid full overlap
            offset = -0.2 #if hem == "left" else 0.1
            x_vals = [idx + offset, idx + offset + 0.4]
            y_vals = [row["VCVS"], row["Cortex"]]
            linestyle = '-' if hem == "left" else '--'
            ax.plot(x_vals, y_vals, color="black", linewidth=1.5, alpha=0.5, linestyle=linestyle)

        # Combine all valid VCVS–Cortex pairs for stat
        vcvs_vals = df_pivot["VCVS"].dropna()
        cortex_vals = df_pivot["Cortex"].dropna()

        if len(vcvs_vals) == len(cortex_vals):
            res = permutation_test(
                (vcvs_vals, cortex_vals),
                statistic=lambda x, y: np.mean(x - y),
                n_resamples=10000,
                alternative='two-sided',
                random_state=42,
            )
            pval = res.pvalue
            symbol = "*" if pval < 0.05 else "n.s."

            # Add single significance marker
            y_max = df_score["correlation"].max()
            ax.text(idx, y_max + 0.05, symbol,
                    ha='center', va='bottom', fontsize=11, color='black',)

    # despine the plot
    sns.despine(trim=True, right=True, top=True)
    plt.xticks(rotation=90)
    plt.ylabel("Correlation with scores")
    plt.title("Correlation of features with scores")
    ax.legend(title="Channel", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.ylim(0.2, 1.0)
    # limit xticks only to first 10 characters of current labels
    #plt.xticks(np.arange(len(cols_plt)), [f[:10] for f in cols_plt])
    pdf_.savefig(plt.gcf())
    plt.close()

pdf_.close()


#####################################################################################

pdf_ = PdfPages("figures/ECoG_SubCortex_correlations_LR_Split.pdf")

for i in np.arange(0, 35, 7):
    cols_plt = cols_all[i:i+7]
    df_plt = df_corr_best.query("score_column in @cols_plt").reset_index(drop=True)
    #df_plt = df_corr_best.query("score_column in @cols_of_interest").reset_index(drop=True)
    plt.figure(figsize=(15, 10))
    order_ = ["VCVS_left", "VCVS_right", "Cortex_left", "Cortex_right"]
    sns.boxplot(data=df_plt , x="score_column", hue="channel", y="correlation", showfliers=False, hue_order=order_)
    sns.swarmplot(data=df_plt, x="score_column", hue="channel", y="correlation", color=".25", dodge=True, legend=False, hue_order=order_)
    plt.xticks(rotation=90)
    plt.ylabel("Correlation with scores")
    plt.title("Correlation of features with scores")
    plt.legend(title="Channel", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    pdf_.savefig(plt.gcf())

    plt.close()
pdf_.close()

