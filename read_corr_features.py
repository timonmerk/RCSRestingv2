import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("correlation_features.csv")


#df_best_max_corr = df.groupby(["subject", "channel", "score_column"])["correlation"].max()
idx = df.groupby(["subject", "channel", "score_column"])["correlation"].idxmax()
df_best = df.loc[idx].reset_index(drop=True)

column = "YBOCS II-Obsessions Sub-score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
plt.figure(figsize=(15, 45))
for c_idx, column in enumerate(df_best["score_column"].unique()):
    plt.subplot(20, 3, c_idx + 1)
    df_sc_column = df_best.query("subject != 'ALL' and channel == 'VCVS_left' and score_column == @column").reset_index()
    sns.barplot(data=df_sc_column, x="correlation", y="subject")
    # write as yticks the feature_name but below \n the subject name
    plt.yticks(ticks=range(len(df_sc_column)), labels=[f"{subj}\n{feat}" for feat, subj in zip(df_sc_column["feature_name"].tolist(), df_sc_column["subject"].tolist())], fontsize=8)
    # write the pvalue right next to the bar
    for i, pval in enumerate(df_sc_column["p_value"].tolist()):
        plt.text(df_sc_column["correlation"].iloc[i], i, f"{pval:.3f}", va='center')
    # turn off right and top spines
    sns.despine(trim=True, right=True, top=True)
    plt.title(column)
plt.tight_layout()
plt.savefig("figures/correlation_features_best.pdf")
plt.show(block=True)

