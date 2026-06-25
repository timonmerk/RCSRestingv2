import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, linear_model
import tqdm_joblib
from xgboost import XGBRegressor, XGBClassifier
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("FAUS_rs/fau_neural_combined.csv")
col_decode = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"

subs = df["subject"].unique()
AU_cols = [col for col in df.columns if col.startswith("FAU_AU")]

### correlation with YBOCS II Total Score
corr_ = []
corr_shuffled = []

sign_list = []

for index in tqdm(range(5000)):
    corr_shuffled_idx = []
    for sub in subs:
        df_sub = df.query("subject == @sub")
        y_true_ = df_sub[col_decode].values
        for AU_col in AU_cols:
            y_pred = df_sub[AU_col].values
            idx_na = np.isnan(y_true_) | np.isnan(y_pred)
            y_true = y_true_[~idx_na]
            y_pred = y_pred[~idx_na]

            # get both from stats.pearsonr
            corr, p_val = stats.pearsonr(y_true, y_pred)
            corr_.append({
                "subject": sub,
                "AU": AU_col,
                "corr": corr,
                "p": p_val,
            })
            np.random.shuffle(y_true)
            corr_shuff, p_val_shuff = stats.pearsonr(y_true, y_pred)
            corr_shuffled_idx.append({ # 
                "subject": sub,
                "AU": AU_col,
                "corr": corr_shuff,
                "p": p_val_shuff,
                "index": index,
            })
    df_fau_corrs_shuffled = pd.DataFrame(corr_shuffled_idx)
    rejected, pvals_corrected, _, _ = multipletests(
        df_fau_corrs_shuffled["p"], alpha=0.05, method="fdr_bh"
    )
    # number of significant correlations after FDR correction
    num_significant = np.sum(rejected)
    sign_list.append({
        "index": index,
        "num_significant": num_significant,
    })
df_corr = pd.DataFrame(corr_)
df_corr_shuff = pd.DataFrame(corr_shuffled)
df_corr_shuff.to_csv("FAUS_rs/fau_neural_decoding_correlation_shuffled.csv", index=False)
df_corr.to_csv("FAUS_rs/fau_neural_decoding_correlation.csv", index=False)


COMPUTE = False

# per_ = []
# for sub in tqdm(subs):
#     for AU_col in AU_cols + ["ALL"]:
#         for model_name in ["Linear", "XGB"]:
def compute_model(df, sub, AU_col, col_decode, model_name):
    col_sel = AU_col if AU_col != "ALL" else AU_cols
    X_train = df.query("subject != @sub")[col_sel]
    X_test = df.query("subject == @sub")[col_sel]
    y_train = df.query("subject != @sub")[col_decode]
    y_test = df.query("subject == @sub")[col_decode]

    model = XGBRegressor() if model_name == "XGB" else linear_model.LinearRegression()
    y_test_na_indexes = y_test.isna()
    y_test = y_test[~y_test_na_indexes]
    X_test = X_test[~y_test_na_indexes]
    y_train_na_indexes = y_train.isna()
    y_train = y_train[~y_train_na_indexes]
    X_train = X_train[~y_train_na_indexes]
    if len(y_train) == 0 or len(y_test) == 0:
        return
    if len(X_train.shape) == 1:
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    corr = np.corrcoef(y_test, y_pred)[0, 1]
    return{
        "subject": sub,
        "model": model_name,
        "correlation": corr,
        "AU": AU_col,
    }

AU_types = ["ALL"] + AU_cols
models = ["Linear", "XGB"]
tasks = [
    delayed(compute_model)(df, sub, AU_col, col_decode, model_name)
    for sub in subs
    for AU_col in AU_types
    for model_name in models
]

if COMPUTE:
    with tqdm_joblib.tqdm_joblib(tqdm(desc="Running models", total=len(tasks))):
        per_ = Parallel(n_jobs=-1)(tasks)
    per__ = [res for res in per_ if res is not None]
    df_results = pd.DataFrame(per__)

    df_results.to_csv("FAUS_rs/fau_neural_decoding_results.csv", index=False)
else:
    df_results = pd.read_csv("FAUS_rs/fau_neural_decoding_results.csv")

df_results.groupby(["AU", "model"])["correlation"].mean().reset_index().sort_values("correlation", ascending=False).head(10)
model_best = "Linear"
AU_best = "FAU_AU_R6"
df_best = df_results.query("model == @model_best and AU == @AU_best")
df_best.to_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/figure_4_FAU/fau_loso_decoding_ybocs_best.csv")

plt.figure(figsize=(12, 8))
for idx_, model in enumerate(models):
    df_model = df_results.query("model == @model")
    # order ascending by mean correlation
    order_ = df_model.groupby("AU")["correlation"].mean().sort_values(ascending=False).index
    plt.subplot(2, 1, idx_ + 1)
    plt.title(f"Model: {model}")
    plt.xlabel("AU Type")
    plt.ylabel("Correlation with YBOCS II Total Score")
    sns.boxplot(x="AU", y="correlation", data=df_model, showmeans=True,
                boxprops=dict(alpha=0.5, facecolor='white', edgecolor='black'), order=order_, showfliers=False)
    sns.swarmplot(x="AU", y="correlation", data=df_model, color=".25", alpha=0.5, order=order_)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("FAUS_rs/fau_loso_neural_decoding_results.pdf")


order_ = df_corr.groupby("AU")["corr"].mean().sort_values(ascending=False).index
plt.figure(figsize=(12, 8))
sns.boxplot(x="AU", y="corr", data=df_corr, showmeans=True,
            boxprops=dict(alpha=0.5, facecolor='white', edgecolor='black'), showfliers=False, order=order_)
sns.swarmplot(x="AU", y="corr", data=df_corr, color=".25", alpha=0.5, order=order_)
plt.title("Correlation of AUs with YBOCS II Total Score")
plt.xlabel("AU Type")
plt.ylabel("Correlation")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("FAUS_rs/fau_neural_decoding_correlation.pdf")