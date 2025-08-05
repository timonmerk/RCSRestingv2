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

def get_date(f, ):
    idx_start = f.find("resting-state")
    idx_start = idx_start + f[idx_start:].find("_") + 1
    idx_sync_or_toggle = f.find("sync") if "sync" in f else f.find("toggle")
    if "BSoff" in f or "BSon" in f:
        idx_start = f.find("BSoff") + len("BSoff") + 1 if "BSoff" in f else f.find("BSon") + len("BSon") + 1
    return f[idx_start : idx_sync_or_toggle-1]

def convert_to_datetime(d):
    if d.startswith("20"):
        dt_ = datetime.strptime(d, "%Y%m%d%H%M%S%f")
    else:
        dt_ = datetime.fromtimestamp(int(d) / 1000)
    # keep only Ymd
    dt_ = dt_.strftime("%Y-%m-%d")
    return dt_


df_ = pd.read_csv("features_prep_combined.csv")
df_loc = df_.query("new_ch == 'VCVS_left'")
col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
df_features = df_loc.groupby(["file", "feature_name"])[[col_score, "feature_value"]].mean().reset_index()
df_features["date"] = df_features["file"].apply(get_date)
df_features["date"] = df_features["date"].apply(convert_to_datetime)
df_features["subject"] = df_features["file"].apply(lambda x: x.split("_")[0])

df_features = df_features.pivot_table(index=["file"], columns="feature_name", values="feature_value").reset_index()
df_features[col_score] = df_loc.groupby("file")[col_score].mean().values
df_features["date"] = df_features["file"].apply(get_date)
df_features["date"] = df_features["date"].apply(convert_to_datetime)
df_features["subject"] = df_features["file"].apply(lambda x: x.split("_")[0])
df_features = df_features.reset_index(drop=True)

cols_keep = list(df_features.columns)[1:-2]

df_features = df_features.groupby(["subject", "date"])[cols_keep].mean().reset_index()


score_thresholds = {
    "aDBS004" : 49 * 0.65,
    "aDBS005" : 46 * 0.65,
    "aDBS007" : 47 * 0.65,
    "aDBS008" : 37 * 0.65,
    "aDBS009" : 45 * 0.65,
    "aDBS010" : 37 * 0.65,
    "aDBS011" : 45 * 0.65,
    "aDBS012" : 46 * 0.65,
}

# add a new column responde that is 1 if the score is above the threshold, else 0
df_features["response"] = 0
for sub, threshold in score_thresholds.items():
    df_features.loc[df_features["subject"] == sub, "response"] = (df_features["YBOCS II Total Score"] > threshold).astype(int)

CLASSIFICATION = True

corrs = []
p_val = []
ba_ = []
true_preds = []
preds = []

def compute_ml(feature_mod: str, CLASSIFICATION: bool = False, model_type: str = "XGB", COMPUTE_PERMUTATION = False):

    per_ = []
    for idx, sub_test in enumerate(df_features["subject"].unique()):
        df_sub = df_features.query("subject == @sub_test")
        for idx_t_sub in range(df_sub.shape[0]):
            X_other_subs 
        X_train = df_features.query("subject != @sub").drop(columns=["subject", col_score, "response", "date"])
        X_test = df_features.query("subject == @sub").drop(columns=["subject", col_score, "response", "date"])

        if feature_mod != "all":
            if feature_mod == "psd":
                cols_keep = ["delta", "theta", "alpha", "beta", "gamma"]
            elif feature_mod == "burst_low_f":
                cols_keep = ["burst_amplitude_alpha", "burst_amplitude_delta", "burst_amplitude_theta"]
            else:
                cols_keep = [c for c in X_train.columns if feature_mod in c]
            X_train = X_train[cols_keep]
            X_test = X_test[cols_keep]
            

        if CLASSIFICATION is False:
            y_test = df_features.query("subject == @sub")[col_score]
            y_train = df_features.query("subject != @sub")[col_score]
        else:
            y_test = df_features.query("subject == @sub")["response"]
            y_train = df_features.query("subject != @sub")["response"]

        y_tr_nan_idx = np.isnan(y_train)
        y_te_nan_idx = np.isnan(y_test)
        X_train = X_train[~y_tr_nan_idx]
        y_train = y_train[~y_tr_nan_idx]
        X_test = X_test[~y_te_nan_idx]
        y_test = y_test[~y_te_nan_idx]
        # XGBoost Regressor
        if CLASSIFICATION is False:
            if model_type == "CatBoost":
                model = CatBoostRegressor(verbose=0, random_seed=42)
            elif model_type == "XGB":
                model = XGBRegressor()
            elif model_type == "Linear":
                model = linear_model.LinearRegression()
        else:
            if model_type == "CatBoost":
                model = CatBoostClassifier(verbose=0, random_seed=42)
            elif model_type == "XGB":
                model = XGBClassifier()
            elif model_type == "Linear":
                model = linear_model.LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        p = None
        if CLASSIFICATION is False:
            corr, p = stats.pearsonr(y_test, y_pred)
            per = corr
        else:
            per = metrics.balanced_accuracy_score(y_test, y_pred)
            if COMPUTE_PERMUTATION:
                n_permutations = 1000
                perm_scores = []
                for _ in range(n_permutations):
                    y_permuted = shuffle(y_test)
                    perm_score = metrics.balanced_accuracy_score(y_permuted, y_pred)
                    perm_scores.append(perm_score)
                perm_scores = np.array(perm_scores)
                p = np.mean(perm_scores >= per)  # p-value is the proportion of

        per_.append({
            #"y_test": y_test,
            #"y_pred": y_pred,
            "per" : per,
            "p_value": p,
            "subject": sub,
            "model": model_type,
            "feature_mod": feature_mod,
            "CLASSIFICATION": CLASSIFICATION,
        })
    return per_

ML_models = ["XGB", "CatBoost", "Linear", ]
feature_mods = ["all", "fooof", "psd", "Sharpwave", "burst", "raw", "Hjorth", "burst_low_f"]

results = []
for model_type in tqdm(ML_models):
    for CLASSIFICATION in [True, False]:
        for feature_mod in feature_mods:
            per_ = compute_ml(feature_mod, CLASSIFICATION, model_type)
            results.append(per_)

df_results = pd.DataFrame(list(np.ravel(results)))
df_results["model_type"] = df_results["model"] + "_" + df_results["feature_mod"]

df_results.to_csv("results_ml.csv", index=False)
CLASSIFICATION_ =False

plt.figure(figsize=(8, 5))
df_plt = df_results.query("CLASSIFICATION == @CLASSIFICATION_").copy()
order_ = df_plt.groupby("model_type")["per"].mean().sort_values(ascending=False).index.tolist()
sns.boxplot(data=df_results.query("CLASSIFICATION == @CLASSIFICATION_"), x="model_type", y="per", order=order_, showmeans=True)
if CLASSIFICATION_:
    plt.axhline(0.5, color="gray", linestyle="--", label="Chance level")
plt.xticks(rotation=90)
sns.swarmplot(data=df_results.query("CLASSIFICATION == @CLASSIFICATION_"), x="model_type", y="per", color=".25", order=order_)
plt.ylabel("Person correlation coefficient") if CLASSIFICATION_ is False else plt.ylabel("Balanced accuracy")
plt.title(f"ML LOSO Classification" if CLASSIFICATION_ else "ML LOSO Regression")

plt.tight_layout()
plt.savefig("figures/decoding/ml_classification.pdf") if CLASSIFICATION_ else plt.savefig("figures/decoding/ml_regression.pdf")
plt.show(block=True)

df_results.query("CLASSIFICATION == False and model_type == 'XGB_burst'")