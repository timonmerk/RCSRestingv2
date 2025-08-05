import pandas as pd
from sklearn import metrics, linear_model
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from scipy import stats
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import numpy as np
import seaborn as sns
from decoder_old import compute_ml
from utils import get_date, convert_to_datetime, preprocess_features

df = pd.read_csv("features_prep_combined.csv")
col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
CLASSIFICATION = False

df_all = preprocess_features(df, col_score)

COMPUTE_BEST = False
if COMPUTE_BEST:
    CLASSIFICATION = False
    loc_test = "VCVS"
    feature_mods = "psd"

    df_X = df_all.drop(columns=[ "file", "loc"])
    features_ = [c for c in df_X.columns if c not in ["subject", "date", col_score, "response"] and loc_test in c]
    features_ += ["subject", "date", col_score, "response"]
    df_X = df_X[features_].reset_index(drop=True)

    res_ = compute_ml(df_X, col_score, feature_mods, CLASSIFICATION,
                      "XGB",
                    COMPUTE_PERMUTATION=False, PLOT_=True,
                    RETURN_ALSO_PREDICTED=True,
                    num_months=3,
                    Z_SCORE_COL_SCORE=True,
                    NORMALIZE_BY_FIRST_VALUE=False
                    )

    subs = pd.DataFrame(res_)["subject"]
    res_rate_pred = []
    res_rate_true = []
    num_months = 3
    for sub_idx, sub in enumerate(subs):
        y_test = res_[sub_idx]["y_test"]
        y_pred = res_[sub_idx]["y_pred"]
        dates = res_[sub_idx]["dates"]
        #dates = np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates])
        first_months = dates < (dates.iloc[0] + pd.DateOffset(months=num_months))
        idx_first_months = np.where(first_months)[0]
        y_test_first_months = y_test[idx_first_months]
        y_pred_first_months = y_pred[idx_first_months]

        last_months = dates >= (dates.iloc[-1] - pd.DateOffset(months=num_months))
        idx_last_months = np.where(last_months)[0]
        y_test_last_months = y_test[idx_last_months]
        y_pred_last_months = y_pred[idx_last_months]

        res_rate_pred.append(np.mean(y_pred_last_months) / np.mean(y_pred_first_months))
        res_rate_true.append(np.mean(y_test_last_months) / np.mean(y_test_first_months))

    plt.figure()
    plt.bar(np.arange(len(subs)), res_rate_pred, label="Predicted Response Rate", alpha=0.5, color="blue")
    plt.bar(np.arange(len(subs)), res_rate_true, label="True Response Rate", alpha=0.5, color="red")
    plt.xticks(np.arange(len(subs)), subs, rotation=90)
    plt.xlabel("Subject")
    plt.ylabel("Response Rate")
    plt.title(f"Response Rate Comparison for {loc_test} with {feature_mods} features")
    plt.legend()
    plt.tight_layout()
    plt.show()

df_res = pd.DataFrame(res_)
plt.figure()
sns.boxplot(y="per", data=df_res, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5), showfliers=False)
plt.title("Non-normed mean: {:.2f} ± {:.2f}".format(df_res["per"].mean(), df_res["per"].std()))
sns.swarmplot(y="per", data=df_res, color=".25", alpha=0.5)
plt.show()

locs = ["SC_L_", "SC_R_", "C_L_1_", "C_L_2_", "C_R_1_", "C_R_2_", "all"]
feature_mod = ["Hjorth", "fft_psd", "fft_only", "Sharpwave", "fooof", "coherence", "burst", "all", "alpha", "beta", "delta", "gamma", "theta", "burst_amplitude", "burst_duration"]
models = ["XGB", "Linear", "RF", "NeuralNet", "SVR_rbf", "SVR_linear"]

# l_per = []
# for model in ["Linear", "XGB"]:  # 
#     for loc_test in tqdm(["VCVS", "VCVS_left", "VCVS_right", "Cortex_left", "Cortex_right", "Cortex", "left", "right", "all"]):
#         for feature_mods in ["all", "fooof", "psd", "Sharpwave", "burst", "raw", "Hjorth", "burst_low_f"]:
#             for NORMALIZE_BY_FIRST_VALUE in [False, True]:
#                 for Z_SCORE_COL_SCORE in [False, True]:
#                     for num_months in [1, 2, 3]:

def run_model_feature_loc_sub(loc, feature, model_name):
    df_X = df_all.drop(columns=[ "file", "loc"])
    if loc != "all":
        features_ = [c for c in df_X.columns if c not in ["subject", "date", col_score, "response"] and loc in c]
        features_ += ["subject", "date", col_score, "response"]
        df_X = df_X[features_].reset_index(drop=True)
    df_per = compute_ml(df_X, col_score, feature_mods, CLASSIFICATION, model_name,
                        COMPUTE_PERMUTATION=False, PLOT_=False,
                        NORMALIZE_BY_FIRST_VALUE=False,
                        Z_SCORE_COL_SCORE=False,
                        num_months=num_months)
    df_per = pd.DataFrame(df_per)
    df_per["loc"] = loc_test
    df_per["model"] = model_name
    df_per["region"] = "VCVS" if "VCVS" in loc else "Cortex" if "Cortex" in loc else "both"
    df_per["hem"] = "left" if "left" in loc else "right" if "right" in loc else "both"
    df_per["feature"] = feature
    df_per["num_months"] = num_months
    df_per["res_pred_all_neg"] = np.sum(df_per["res_diff_pred"] < 0)

    return df_per

run_model_feature_loc_sub("VCVS_L", "psd", "XGB")

df_per_all = pd.concat(l_per, axis=0).reset_index()
df_per_all.to_csv("decoder_performance_including_diff.csv", index=False)

df_per_all["model_loc_feature"] = df_per_all["model"] + "_" + df_per_all["loc"] + "_" + df_per_all["feature_mod"]

df_per_all.groupby("model_loc_feature")["per"].mean().sort_values(ascending=False).reset_index()

loc_test="VCVS"
features_ = [c for c in df_X.columns if c not in ["subject", col_score, "response"] and loc_test in c]
features_ += ["subject", "date", col_score, "response"]
df_X = df_all.drop(columns=[ "file", "loc"])
df_X = df_X[features_].reset_index(drop=True)
compute_ml(df_X, col_score, "psd", True, "XGB", COMPUTE_PERMUTATION=False, PLOT_=True)


plt.figure()
sns.boxplot(data=df_per_all, x="feature_mod", y="per", hue="loc", showfliers=False, showmeans=True, boxprops=dict(alpha=0.5))
sns.swarmplot(data=df_per_all, x="feature_mod", y="per", hue="loc", dodge=True, color="black", alpha=0.5, legend=False)
plt.xticks(rotation=90)
plt.ylabel("Pearson's r")
plt.tight_layout()

# df_per_all.groupby(["feature_mod", "loc"])["per"].mean().reset_index().sort_values("per", ascending=False)
# get the first value for each loc
df_best_per_loc = df_per_all.groupby(["loc", "feature_mod"])["per"].mean().reset_index().sort_values("per", ascending=False).groupby("loc").first().reset_index()
df_best_per_loc["loc_feature"] = df_best_per_loc["loc"] + "_" + df_best_per_loc["feature_mod"]
# sort by per
df_best_per_loc = df_best_per_loc.sort_values("per", ascending=False).reset_index(drop=True)
# select only the combination of loc and feature_mod
df_sel = []
for row in df_best_per_loc.itertuples():
    loc = row.loc
    feature_mod = row.feature_mod
    df_sel.append(df_per_all.query("loc == @loc and feature_mod == @feature_mod"))
df_per_best = pd.concat(df_sel, axis=0).reset_index(drop=True)
#df_per_best["loc_feature"] = df_per_best["loc"] + "_" + df_per_best["feature_mod"]

# sort by per
df_per_best = df_per_best.sort_values("per", ascending=False).reset_index(drop=True)
# show those in a boxplot
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df_per_best,
    x="loc_feature",
    y="per",
    order=df_best_per_loc["loc_feature"].tolist()[::-1],
    showfliers=False,
    showmeans=True,
    boxprops=dict(alpha=0.5)
)
sns.swarmplot(
    data=df_per_best,
    x="loc_feature",
    y="per",
    order=df_best_per_loc["loc_feature"].tolist()[::-1],
    dodge=True,
    color="black",
    alpha=0.5,
    legend=False
)
plt.xticks(rotation=90)
plt.ylabel("Pearson's r")
plt.tight_layout()

plt.figure()
for idx, loc in enumerate(["VCVS_left", "VCVS_right", "Cortex_left", "Cortex_right", "VCVS", "Cortex", "left", "right", "all"]):
    plt.subplot(3, 3, idx + 1)
    plt.title(loc)
    sns.boxplot(
        data=df_per_all.query("loc == @loc"),
        x="feature_mod", y="per")
    sns.swarmplot(
        data=df_per_all.query("loc == @loc"),
        x="feature_mod", y="per",
        dodge=True,
        color="black",
        alpha=0.5,
        legend=False
    )
plt.tight_layout()

# the mean best feature_mod per loc
df_per_mean = df_per_all.groupby(["loc", "feature_mod"])["per"].mean().reset_index()
# get the best feature_mod per loc
best_feature_mods = df_per_mean.loc[df_per_mean.groupby("loc")["per"].idxmax()]
best_feature_mods["loc_feature"] = best_feature_mods["loc"] + "_" + best_feature_mods["feature_mod"]
best_feature_mods = best_feature_mods.sort_values("per")


df_best = []
for loc in best_feature_mods["loc"].unique():
    df_loc = df_per_all.query("loc == @loc")
    feature_mod = best_feature_mods.query("loc == @loc")["feature_mod"].values[0]
    df_loc = df_loc.query("feature_mod == @feature_mod")
    df_best.append(df_loc)
df_per_best = pd.concat(df_best, axis=0).reset_index(drop=True)

# show those in a boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df_per_best,
    x="model_loc_feature",
    y="per",
    #order=best_feature_mods["loc_feature"].tolist(),
    showfliers=False,
    showmeans=True,
    boxprops=dict(alpha=0.5)
)
sns.swarmplot(
    data=df_per_best,
    x="model_loc_feature",
    y="per",
    #order=best_feature_mods["loc_feature"].tolist(),
    dodge=True,
    color="black",
    alpha=0.5,
    legend=False
)
plt.xticks(rotation=90)
plt.ylabel("Pearson's r")
plt.tight_layout()

