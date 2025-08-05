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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy import stats
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


df = pd.read_csv("features_prep_combined.csv")
col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
CLASSIFICATION = False

df_all = preprocess_features(df, col_score)

CLASSIFICATION = False
loc_test = "VCVS"
feature_mod = "psd"
model_type = "XGB"
num_months = 3
RETURN_ALSO_PREDICTED = True

df_X = df_all.drop(columns=[ "file", "loc"])
features_ = [c for c in df_X.columns if c not in ["subject", "date", col_score, "response"] and loc_test in c] # 
features_ += ["subject", "date", col_score, "response"]
df_X = df_X[features_].reset_index(drop=True)

df_features = df_X.copy()  # Replace with your actual DataFrame
# Normalize score within each subject
# for sub in df_features["subject"].unique():
#     mask = df_features["subject"] == sub
#     mean_score = df_features.loc[mask, col_score].mean()
#     std_score = df_features.loc[mask, col_score].std()
#     # Uncomment to normalize:
#     # df_features.loc[mask, col_score] = (df_features.loc[mask, col_score] - mean_score) / std_score

score_thresholds = {
    "aDBS004": 49 * 0.65,
    "aDBS005": 46 * 0.65,
    "aDBS007": 47 * 0.65,
    "aDBS008": 37 * 0.65,
    "aDBS009": 45 * 0.65,
    "aDBS010": 37 * 0.65,
    "aDBS011": 45 * 0.65,
    "aDBS012": 46 * 0.65,
}

# Add binary classification column
df_features = df_X.copy()
col_score = "YBOCS II Total Score"
df_features["response"] = 0
for sub, threshold in score_thresholds.items():
    mask = df_features["subject"] == sub
    df_features.loc[mask, "response"] = (df_features.loc[mask, col_score] > threshold).astype(int)

# Drop rows with NaNs
df_features = df_features.dropna(subset=["response", "date"])

X = df_features.drop(columns=["subject", col_score, "response", "date"])
y = df_features["response"]
subjects = df_features["subject"].values
dates = pd.to_datetime(df_features["date"])

# Optional feature filtering
if feature_mod != "all":
    if feature_mod == "psd":
        cols_keep = ["delta", "theta", "alpha", "beta", "gamma",
                     "burst_amplitude_alpha", "burst_amplitude_delta", "burst_amplitude_theta"]
    elif feature_mod == "burst_low_f":
        cols_keep = ["burst_amplitude_alpha", "burst_amplitude_delta", "burst_amplitude_theta"]
    elif feature_mod == "theta_only":
        cols_keep = ["theta"]
    else:
        cols_keep = [c for c in X.columns if feature_mod in c]
    X = X[[c for c in X.columns if any(c.endswith(k) for k in cols_keep)]]

# Remove any NaNs
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]
subjects = subjects[mask]
dates = dates[mask]

# Single train/test split across all subjects
from sklearn.model_selection import train_test_split

# Combine all into one DataFrame for consistent splitting
df_all_features = X.copy()
df_all_features["response"] = y
df_all_features["subject"] = subjects
df_all_features["date"] = dates

# Do a stratified split if class balance is important; otherwise drop stratify
train_df, test_df = train_test_split(
    df_all_features, test_size=0.33, random_state=42, shuffle=True, stratify=df_all_features["response"]
)

# Separate back into components
X_train = train_df.drop(columns=["response", "subject", "date"])
y_train = train_df["response"]
X_test = test_df.drop(columns=["response", "subject", "date"])
y_test = test_df["response"]
sub_test = test_df["subject"]
dates_test = pd.to_datetime(test_df["date"])

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Collect results per subject in test set
results = []
for sub in np.unique(sub_test):
    mask = sub_test == sub
    if mask.sum() < 2:
        continue
    yt = y_test[mask]
    yp = y_pred[mask]
    yp_prob = y_prob[mask]
    dt = dates_test[mask]

    acc = accuracy_score(yt, yp)
    ba = metrics.balanced_accuracy_score(yt, yp)

    first_months = dt < (dt.min() + pd.DateOffset(months=num_months))
    last_months = dt >= (dt.max() - pd.DateOffset(months=num_months))

    res_diff_pred = np.mean(yp_prob[last_months]) - np.mean(yp_prob[first_months])
    res_diff_true = np.mean(yt[last_months]) - np.mean(yt[first_months])
    diff_first_last_true_pred = res_diff_pred - res_diff_true

    results.append({
        "subject": sub,
        "accuracy": acc,
        "ba" : ba,
        "y_test": yt if RETURN_ALSO_PREDICTED else None,
        "y_pred": yp if RETURN_ALSO_PREDICTED else None,
        "y_prob": yp_prob if RETURN_ALSO_PREDICTED else None,
        "dates": dt if RETURN_ALSO_PREDICTED else None,
        "res_diff_pred": res_diff_pred,
        "res_diff_true": res_diff_true,
        "diff_first_last_true_pred": diff_first_last_true_pred,
        "feature_mod": f"XGB_{feature_mod}"
    })

df_all = pd.DataFrame(results)