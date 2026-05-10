import numpy as np
import pandas as pd
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import permutation_test

X_ = pd.read_csv("features_prep_combined_wide.csv")
X_["date"] = pd.to_datetime(X_["date"])
first_date_per_sub = X_.groupby("subject")["date"].min().reset_index()
# asign new coln "time_since_first_date" for each subject
X_ = X_.merge(first_date_per_sub, on="subject", suffixes=("", "_first"))
X_["time_since_first_date"] = (X_["date"] - X_["date_first"]).dt.days

cols_test = ["YBOCS II Total Score", "subject", "time_since_first_date", 'Category 1:Concerns about Germs and Cotamination- Subscale Score',
             'Category 2: Concerns about being Responsible for Harm, Injury, or Bad Luck- Subscale Scale',
             'Category 3: Unacceptable Thoughts-Subscale Score', 'Category 4: Concerns about Symmetry, Completeness',
             'Category 5: Sexually Intrusive Thoughts- Subscale Score', 'Category 6: Intrusive Violent Thoughts- Subscale Score',
             'Category 7: Immoral and Scrupulous Thoughts- Subscale Score', 
            'BAI-Total Score', 'HDRS Total Score', 'BAI-Total Score', 'YBOCS II-Obsessions Sub-score', 'YBOCS II-Compulsions Sub-score', 'YMRS Total Score', ]

cols_test = ["YBOCS II Total Score", "subject", "time_since_first_date", ]

def run_permanova(X_, col_):
    # select only rows that start with C_ or SC_
    if CORTEX_ONLY:
        X = X_[[c for c in X_.columns if c.startswith("C_") or c == col_]]
        # remove subjects aDBS004, aDBS005, aDBS007
        #X = X[~X["subject"].isin(["aDBS004", "aDBS005", "aDBS007"])]
        # drop rows where "C_L_1_RawHjorth_Activity" is NaN
        X = X.dropna(subset=["C_L_1_RawHjorth_Activity"])
    else:
        X = X_[[c for c in X_.columns if c.startswith("SC_") or c == col_]]
    
    # remove cols with NaN but keep col_ if it has NaN
    X = X.dropna(axis=1, how='any')
    X[col_] = X_[col_]
    # remove rows with NaN in col_
    X = X.dropna(subset=[col_])


    y = X[col_].values
    # discretize y into 8 bins
    if col_ != "subject":
        y = pd.qcut(y, q=8, labels=False, duplicates='drop')
    X = X.drop(columns=[col_]).values

    # score: continuous array (same length as samples)
    distY = squareform(pdist(X, metric="euclidean"))
    dm = DistanceMatrix(distY)

    metadata = pd.DataFrame({"score": y}, index=range(len(X)))
    results = permanova(dm, y, permutations=1000)
    results["col"] = col_
    return results

res_p = []
CORTEX_ONLY = False

for col_ in cols_test:
    res = run_permanova(X_, col_)
    res["across_patients"] = True
    res_p.append(res)
    for sub in X_["subject"].unique():
        if col_ == "subject":
            continue
        res = run_permanova(X_[X_["subject"] == sub], col_)
        res["across_patients"] = False
        res_p.append(res)

df_ = pd.DataFrame(res_p)
# sort df_ by "test statistic" descending
df_ = df_.sort_values(by="test statistic", ascending=True)

plt.figure(figsize=(7, 5))
plt.subplot(1, 2, 1)
df_plt = df_[df_["across_patients"] == True]
plt.barh(df_plt["col"], df_plt["test statistic"])
plt.xlabel("PERMANOVA test statistic")
plt.title("PERMANOVA across patients")
# plot the p-value as text on the bars
for i, (stat, p) in enumerate(zip(df_plt["test statistic"], df_plt["p-value"])):
    plt.text(stat, i, f"p={p:.3f}", va='center', ha='left')
plt.subplot(1, 2, 2)
df_plt = df_[df_["across_patients"] == False]
sns.boxplot(x="col", y="test statistic", data=df_plt, showfliers=False, showmeans=True)
sns.stripplot(x="col", y="test statistic", data=df_plt, color='black', alpha=0.5)
# connect the points for each patient with a line
for sub in X_["subject"].unique():
    sub_data = df_plt[df_plt["col"] == "subject"]
    sub_stat = sub_data[sub_data["col"] == "subject"]["test statistic"].values
    if len(sub_stat) > 0:
        plt.plot([0, 1], [sub_stat[0], sub_stat[0]], color='gray', alpha=0.5)
# run a scipy permutation test to compare YBOCS II Total Score vs time_since_first_date
y1 = df_plt[df_plt["col"] == "YBOCS II Total Score"]["test statistic"].values
y2 = df_plt[df_plt["col"] == "time_since_first_date"]["test statistic"].values
res = permutation_test((y1, y2), random_state=0, statistic=lambda x, y: np.mean(x) - np.mean(y), n_resamples=5000)
plt.xticks(rotation=90)
plt.xlabel("PERMANOVA test statistic")
plt.title(f"PERMANOVA within patients \np={res.pvalue:.3f}")
plt.tight_layout()
plt.savefig("figures/permanova_results.pdf")
plt.show()
