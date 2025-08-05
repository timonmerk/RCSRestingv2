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

compute_ml(df_X, col_score, feature_mod, CLASSIFICATION,
           model_type,
           COMPUTE_PERMUTATION=False, PLOT_=True,
           RETURN_ALSO_PREDICTED=RETURN_ALSO_PREDICTED,
           num_months=num_months,
           Z_SCORE_COL_SCORE=True,
           NORMALIZE_BY_FIRST_VALUE=False
           )

compute_ml(df_X, col_score, feature_mod, CLASSIFICATION,
           model_type,
           COMPUTE_PERMUTATION=False, PLOT_=True,
           RETURN_ALSO_PREDICTED=RETURN_ALSO_PREDICTED,
           num_months=num_months,
           Z_SCORE_COL_SCORE=False,
           NORMALIZE_BY_FIRST_VALUE=False
           )

df_features = df_X.copy()

for sub in df_features["subject"].unique():
    df_features_sub = df_features.query("subject == @sub")
    mean_score = df_features_sub[col_score].mean()
    std_score = df_features_sub[col_score].std()
    df_features.loc[df_features["subject"] == sub, col_score] = (df_features_sub[col_score] - mean_score) / std_score

    per_ = []
    per_ba = None
    for idx, sub in enumerate(df_features["subject"].unique()):

        X_train = df_features.query("subject != @sub").drop(columns=["subject", col_score, "response", "date"])
        X_test = df_features.query("subject == @sub").drop(columns=["subject", col_score, "response", "date"])

        if feature_mod != "all":
            if feature_mod == "psd":
                cols_keep = ["delta", "theta", "alpha", "beta", "gamma", "burst_amplitude_alpha", "burst_amplitude_delta", "burst_amplitude_theta"]
            elif feature_mod == "burst_low_f":
                cols_keep = ["burst_amplitude_alpha", "burst_amplitude_delta", "burst_amplitude_theta"]
            elif feature_mod == "theta_only":
                cols_keep = ["theta"]
            else:
                cols_keep = [c for c in X_train.columns if feature_mod in c]
            
            # get columns that end with cols_keep
            cols_keep_ = [c for c in X_train.columns if any(c.endswith(k) for k in cols_keep)]
            X_train = X_train[cols_keep_]
            X_test = X_test[cols_keep_]

        if CLASSIFICATION is False:
            y_test = df_features.query("subject == @sub")[col_score]
            y_train = df_features.query("subject != @sub")[col_score]
  
        dates = pd.to_datetime(df_features.query("subject == @sub")["date"])

        y_tr_nan_idx = np.isnan(y_train)
        y_te_nan_idx = np.isnan(y_test)
        X_train = X_train[~y_tr_nan_idx]
        y_train = y_train[~y_tr_nan_idx]
        X_test = X_test[~y_te_nan_idx]
        y_test = y_test[~y_te_nan_idx]
        dates = dates[~y_te_nan_idx]

        model = XGBRegressor()


        if model_type == "Linear":
            # remove NaN rows
            idx_nan_all = X_train.index[X_train.isna().any(axis=1)]
            X_train = X_train.drop(index=idx_nan_all)
            y_train = y_train.drop(index=idx_nan_all)
            #y_train = np.delete(y_train, idx_nan_all, axis=0)
            idx_nan_test = X_test.index[X_test.isna().any(axis=1)]
            X_test = X_test.drop(index=idx_nan_test)
            y_test = y_test.drop(index=idx_nan_test)
            #y_test = np.delete(y_test, idx_nan_all, axis=0)
            #dates = np.array([dates[i] for i in range(len(dates)) if i not in idx_nan_test])
            dates = dates.drop(index=idx_nan_test)

            if CLASSIFICATION:
                y_test_ybocs2_total = y_test_ybocs2_total.drop(index=idx_nan_test)
            if y_test.shape[0] == 0:
                #print(f"Skipping {sub} due to empty test set")
                continue
        y_test = y_test.values
        y_train = y_train.values
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        p = None
        if CLASSIFICATION is False:
            corr, p = stats.pearsonr(y_test, y_pred)
            per = corr

        first_months = dates < (dates.iloc[0] + pd.DateOffset(months=num_months))
        #idx_first_months = np.where(first_months)[0]
        y_test_first_months = y_test[first_months]
        y_pred_first_months = y_pred[first_months]

        last_months = dates >= (dates.iloc[-1] - pd.DateOffset(months=num_months))
        #idx_last_months = np.where(last_months)[0]
        y_test_last_months = y_test[last_months]
        y_pred_last_months = y_pred[last_months]

        res_diff_pred = np.mean(y_pred_last_months) - np.mean(y_pred_first_months)
        res_diff_true = np.mean(y_test_last_months) - np.mean(y_test_first_months)
        diff_first_last_true_pred = res_diff_pred - res_diff_true

        per_.append({
            "y_test": y_test if RETURN_ALSO_PREDICTED else None,
            "y_pred": y_pred if RETURN_ALSO_PREDICTED else None,
            "dates" : dates if RETURN_ALSO_PREDICTED else None,
            "per" : per,
            "p_value": p,
            "subject": sub,
            "feature_mod": "XGB_psd",
            "res_diff_pred": res_diff_pred,
            "res_diff_true": res_diff_true,
            "diff_first_last_true_pred": diff_first_last_true_pred,
        })

        per_.append({
            "y_test": y_test if RETURN_ALSO_PREDICTED else None,
            "y_pred": y_pred if RETURN_ALSO_PREDICTED else None,
            "dates" : dates if RETURN_ALSO_PREDICTED else None,
            "per" : np.corrcoef(X_test["VCVS_left_theta"], y_test)[0, 1],
            "p_value": p,
            "subject": sub,
            "feature_mod": "theta",
            "res_diff_pred": res_diff_pred,
            "res_diff_true": res_diff_true,
            "diff_first_last_true_pred": diff_first_last_true_pred,
        })

df_all = pd.DataFrame(per_)

mean_theta = df_all[df_all["feature_mod"] == "theta"]["per"].mean()
std_theta = df_all[df_all["feature_mod"] == "theta"]["per"].std()
mean_xgb = df_all[df_all["feature_mod"] == "XGB_psd"]["per"].mean()
std_xgb = df_all[df_all["feature_mod"] == "XGB_psd"]["per"].std()
plt.figure(figsize=(3, 5))
sns.boxplot(x="feature_mod", y="per", data=df_all, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5))
sns.swarmplot(x="feature_mod", y="per", data=df_all, dodge=False, color=".25", alpha=0.5)
plt.title(f"corr XGB: {mean_xgb:.2f} ± {std_xgb:.2f}\ntheta: {mean_theta:.2f} ± {std_theta:.2f}")
plt.xlabel("Feature Modality")
plt.ylabel("Pearson Correlation Coefficient")
plt.tight_layout()

# plot the featureimportance of the XGB model

plt.figure()
plt.barh(X_train.columns, model.feature_importances_, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance of XGB Model')
plt.tight_layout()

import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test) 

shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
mean_effect = shap_df.mean().sort_values()

shap.plots.waterfall(shap_values[5])
plt.tight_layout()

plt.figure(figsize=(10, 6))
# plot the mean effect of each feature
mean_effect.plot(kind='barh', color='skyblue')
plt.xlabel('Mean SHAP Value')
plt.title('Mean SHAP Values for Features')
plt.tight_layout()