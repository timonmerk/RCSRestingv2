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

# Prepare features, targets, dates
X = df_features.drop(columns=["subject", col_score, "response", "date"])
y = df_features[col_score]
dates = pd.to_datetime(df_features["date"])
subjects = df_features["subject"]

# Feature selection
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

# Drop NaNs
mask = ~(X.isna().any(axis=1) | y.isna())
X, y, dates, subjects = X[mask], y[mask], dates[mask], subjects[mask]

# Global train/test split
X_train, X_test, y_train, y_test, dates_train, dates_test, subjects_train, subjects_test = train_test_split(
    X, y, dates, subjects, test_size=0.34, random_state=42, shuffle=True
)

# Train global model
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Collect test-set results per subject
results = []

for sub in subjects_test.unique():
    mask = subjects_test == sub

    if np.sum(mask) < 2:
        continue  # Need at least 2 samples for correlation

    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    dates_sub = dates_test[mask]

    corr, p = stats.pearsonr(y_test_sub, y_pred_sub)

    first_months = dates_sub < (dates_sub.min() + pd.DateOffset(months=num_months))
    last_months = dates_sub >= (dates_sub.max() - pd.DateOffset(months=num_months))

    res_diff_pred = np.mean(y_pred_sub[last_months]) - np.mean(y_pred_sub[first_months])
    res_diff_true = np.mean(y_test_sub[last_months]) - np.mean(y_test_sub[first_months])
    diff_first_last_true_pred = res_diff_pred - res_diff_true

    results.append({
        "subject": sub,
        "y_test": y_test_sub if RETURN_ALSO_PREDICTED else None,
        "y_pred": y_pred_sub if RETURN_ALSO_PREDICTED else None,
        "dates": dates_sub if RETURN_ALSO_PREDICTED else None,
        "per": corr,
        "p_value": p,
        "feature_mod": f"XGB_{feature_mod}",
        "res_diff_pred": res_diff_pred,
        "res_diff_true": res_diff_true,
        "diff_first_last_true_pred": diff_first_last_true_pred,
    })

df_all = pd.DataFrame(results)



# Assuming df_all is your DataFrame with per-subject results
subjects = df_all["subject"].unique()
num_subjects = len(subjects)

# Determine subplot grid size
cols = 4
rows = int(np.ceil(num_subjects / cols))

fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=False)
axs = axs.flatten()

for i, sub in enumerate(subjects):
    ax = axs[i]
    df_sub = df_all[df_all["subject"] == sub].iloc[0]  # assuming one row per subject

    if df_sub["y_test"] is not None and df_sub["y_pred"] is not None:
        dates = df_sub["dates"]
        y_test = df_sub["y_test"]
        y_pred = df_sub["y_pred"]

        # Convert to NumPy arrays and sort by date
        dates = np.array(dates)
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        sort_idx = np.argsort(dates)
        dates_sorted = dates[sort_idx]
        y_test_sorted = y_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        ax.plot(dates_sorted, y_test_sorted, label="True", marker='o')
        ax.plot(dates_sorted, y_pred_sorted, label="Predicted", marker='x')
        ax.set_title(f"Subject {sub}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Score")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

# Hide unused subplots
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3))
sns.boxplot(x="feature_mod", y="per", data=df_all, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5))
sns.swarmplot(x="feature_mod", y="per", data=df_all, dodge=False, color=".25", alpha=0.5)
plt.title(f"corr XGB: {df_all['per'].mean():.2f} ± {df_all['per'].std():.2f}")
plt.xlabel("Feature Modality")
plt.ylabel("Pearson Correlation Coefficient")
plt.tight_layout()