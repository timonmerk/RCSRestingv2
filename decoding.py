import pandas as pd
from sklearn import metrics, linear_model
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from scipy import stats
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime

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
    return dt_


df_ = pd.read_csv("features_prep_combined.csv")
df_loc = df_.query("new_ch == 'VCVS_left'")
col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
df_features = df_loc.groupby(["subject", "file", "feature_name"])[[col_score, "feature_value"]].mean().reset_index()
df_features["date"] = df_features["file"].apply(get_date)
df_features["date"] = df_features["date"].apply(convert_to_datetime)

df_features = df_features.pivot_table(index=["subject", "file"], columns="feature_name", values="feature_value").reset_index()
df_features[col_score] = df_loc.groupby(["subject", "file"])[col_score].mean().values

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

for idx, sub in tqdm(enumerate(df_features["subject"].unique())):

    X_train = df_features.query("subject != @sub").drop(columns=["subject", "file", col_score, "response"])
    X_test = df_features.query("subject == @sub").drop(columns=["subject", "file", col_score, "response"])

    if CLASSIFICATION is False:
        y_test = df_features.query("subject == @sub")[col_score]
        y_train = df_features.query("subject != @sub")[col_score]
    else:
        y_test = df_features.query("subject == @sub")["response"]
        y_train = df_features.query("subject != @sub")["response"]
    # XGBoost Regressor
    if CLASSIFICATION is False:
        model = XGBRegressor()
        model = linear_model.LinearRegression()
        model = CatBoostRegressor(verbose=0, random_seed=42)
    else:
        model = XGBClassifier(class_weights="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if CLASSIFICATION is False:
        corr, p = stats.pearsonr(y_test, y_pred)
        corrs.append(corr)
        p_val.append(p)
    else:
        ba_.append(metrics.balanced_accuracy_score(y_test, y_pred))

    plt.figure()
    plt.plot(y_test.values, label="True Values", marker='o')
    plt.plot(y_pred, label="Predicted Values", marker='x')
    plt.title(f"{sub}")
    plt.legend()
    plt.show()



