import pandas as pd
from sklearn import metrics, linear_model, discriminant_analysis
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy import stats
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import numpy as np
import seaborn as sns

def compute_ml(df_features : pd.DataFrame, col_score: str,
               feature: str,
               CLASSIFICATION: bool = False,
               model_type: str = "XGB",
               loc: str = "SC_L_",
               COMPUTE_PERMUTATION = False,
               PLOT_ = False, RETURN_ALSO_PREDICTED = False, 
               NORMALIZE_BY_FIRST_VALUE = False,
               Z_SCORE_COL_SCORE = False,
               num_months: int = 3):
    if PLOT_:
        plt.figure(figsize=(15, 6))

    
    if NORMALIZE_BY_FIRST_VALUE:
        for sub in df_features["subject"].unique():
            dates_ = df_features.query("subject == @sub")["date"].tolist()
            cols_norm = [c for c in df_features.columns if c not in ["subject", "date", col_score, "response", "file", "loc"]]
            date_idx = 0
            date_first = df_features.query("subject == @sub")["date"].iloc[date_idx]
            while np.nan_to_num(df_features.query("subject == @sub and date == @date_first")[cols_norm].values.sum()) == 0:
                date_idx += 1
                if date_idx >= len(dates_):
                    print(f"Warning: No valid first date found for subject {sub}. Skipping normalization.")
                    break
                date_first = dates_[date_idx]

            baseline = df_features.query("subject == @sub and date == @date_first")[cols_norm].iloc[0]
            df_features.loc[(df_features["subject"] == sub) & (df_features["date"] != date_first), cols_norm] -= baseline
            df_features = df_features.query("not (subject == @sub and date == @date_first)").reset_index(drop=True)

    
    if Z_SCORE_COL_SCORE:
        for sub in df_features["subject"].unique():
            df_features_sub = df_features.query("subject == @sub")
            mean_score = df_features_sub[col_score].mean()
            std_score = df_features_sub[col_score].std()
            df_features.loc[df_features["subject"] == sub, col_score] = (df_features_sub[col_score] - mean_score) / std_score


    per_ = []
    per_ba = None
    for idx, sub in enumerate(df_features["subject"].unique()):

        X_train = df_features.query("subject != @sub").drop(columns=["subject", col_score, "date"])
        X_test = df_features.query("subject == @sub").drop(columns=["subject", col_score, "date"])

        if loc != "all":
            cols_use_loc = [col for col in X_train.columns if col.startswith(loc)]
        else:
            cols_use_loc = list(X_train.columns)
        
        X_train_loc = X_train[cols_use_loc]
        X_test_loc = X_test[cols_use_loc]

        if feature != "all":
            if feature != "fft_only":
                if feature == "theta" or feature == "alpha" or feature == "beta" or feature == "gamma" or feature == "delta":
                    cols_use_feature = [col for col in X_train_loc.columns if feature in col and "fft" not in col and "burst" not in col]
                else:
                    cols_use_feature = [col for col in X_train_loc.columns if feature in col]
            else:
                cols_use_feature = [col for col in X_train_loc.columns if "fft" in col and "psd" not in col]
            X_train_feature = X_train_loc[cols_use_feature]
            X_test_feature = X_test_loc[cols_use_feature]
        else:
            X_train_feature = X_train_loc
            X_test_feature = X_test_loc
            
        if X_train_feature.columns.empty or float(X_test_feature.sum().sum()) == 0:
            return None

        if CLASSIFICATION is False:
            y_test = df_features.query("subject == @sub")[col_score]
            y_train = df_features.query("subject != @sub")[col_score]
        else:
            y_test = df_features.query("subject == @sub")["response"]
            y_train = df_features.query("subject != @sub")["response"]
            y_test_ybocs2_total = df_features.query("subject == @sub")['YBOCS II Total Score']
        
        dates = pd.to_datetime(df_features.query("subject == @sub")["date"])

        y_tr_nan_idx = np.isnan(y_train)
        y_te_nan_idx = np.isnan(y_test)
        X_train = X_train[~y_tr_nan_idx]
        y_train = y_train[~y_tr_nan_idx]
        X_test = X_test[~y_te_nan_idx]
        y_test = y_test[~y_te_nan_idx]
        dates = dates[~y_te_nan_idx]

        if CLASSIFICATION:
            y_test_ybocs2_total = y_test_ybocs2_total[~y_te_nan_idx]
        # XGBoost Regressor
        if CLASSIFICATION is False:
            if model_type == "CatBoost":
                model = CatBoostRegressor(verbose=0, random_seed=42)
            elif model_type == "XGB":
                model = XGBRegressor()
            elif model_type == "Linear":
                model = linear_model.LinearRegression()
            elif model_type == "MLP":
                model = MLPRegressor(random_state=42, max_iter=1000)
            elif model_type == "SVR_linear":
                model = SVR(kernel='linear', C=1.0, epsilon=0.1)
            elif model_type == "SVR_rbf":
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            elif model_type == "RF":
                model = RandomForestRegressor()
        else:
            if model_type == "CatBoost":
                model = CatBoostClassifier(verbose=0, random_seed=42)
            elif model_type == "XGB":
                model = XGBClassifier()
            elif model_type == "Linear":
                model = linear_model.LogisticRegression()
        if model_type != "XGB":
            # remove NaN rows
            idx_nan_all = X_train_feature.index[X_train_feature.isna().any(axis=1)]
            X_train_feature = X_train_feature.drop(index=idx_nan_all)
            y_train = y_train.drop(index=idx_nan_all)
            #y_train = np.delete(y_train, idx_nan_all, axis=0)
            idx_nan_test = X_test_feature.index[X_test_feature.isna().any(axis=1)]
            X_test_feature = X_test_feature.drop(index=idx_nan_test)
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
        model.fit(X_train_feature, y_train)
        y_pred = model.predict(X_test_feature)
        if PLOT_:
            if CLASSIFICATION:
                y_pred_proba = model.predict_proba(X_test_feature)
                y_pred_proba = pd.Series(y_pred_proba[:, 1] * 40).rolling(window=2, center=True).mean()
                y_true = y_test*40
                score = np.round(metrics.balanced_accuracy_score(y_test, y_pred), 2)
            else:
                y_pred = model.predict(X_test_feature)
                y_true = y_test
                score = np.round(stats.pearsonr(y_test, y_pred)[0], 2)
            plt.subplot(2, 4, idx + 1)
            plt.plot(y_true, label="True")
            if CLASSIFICATION:
                plt.plot(y_pred_proba, label="Predicted Proba")
                plt.plot(y_test_ybocs2_total.values, label="YBOCS II Total Score")
            else:
                plt.plot(y_pred, label="Predicted")
            #plt.plot(y_test_ybocs2_total.values, label="YBOCS II Total Score")
            plt.title(f"sub: {sub} per: {score}")
            dates_plt = df_features.query("subject == @sub")["date"].tolist()
            plt.xticks(ticks=range(len(dates_plt)), labels=dates_plt, rotation=90)
            plt.legend()
            #plt.show(block=True)
            if idx == 0:
                plt.legend()

        p = None
        if CLASSIFICATION is False:
            corr, p = stats.pearsonr(y_test, y_pred)
            per = corr
        else:
            per = metrics.balanced_accuracy_score(y_test, y_pred)
            per_ba = per
            y_pred_proba = model.predict_proba(X_test_feature) if CLASSIFICATION else None
            pr__roll = pd.Series(y_pred_proba[:, 1] * 40).rolling(window=2, center=True).mean()
            idx_both_not_na = ~np.isnan(y_test_ybocs2_total.values) & ~np.isnan(pr__roll.values)
            y_test_ybocs2_total = y_test_ybocs2_total[idx_both_not_na]
            pr__roll = pr__roll[idx_both_not_na]
            rho, per = stats.pearsonr(y_test_ybocs2_total.values, pr__roll.values)
            if COMPUTE_PERMUTATION:
                n_permutations = 1000
                perm_scores = []
                for _ in range(n_permutations):
                    y_permuted = shuffle(y_test)
                    perm_score = metrics.balanced_accuracy_score(y_permuted, y_pred)
                    perm_scores.append(perm_score)
                perm_scores = np.array(perm_scores)
                p = np.mean(perm_scores >= per)  # p-value is the proportion of

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
            "per_ba_class" : per_ba,
            "subject": sub,
            "model": model_type,
            "feature": feature,
            "CLASSIFICATION": CLASSIFICATION,
            "res_diff_pred": res_diff_pred,
            "res_diff_true": res_diff_true,
            "diff_first_last_true_pred": diff_first_last_true_pred,
        })
    if PLOT_:
        plt.tight_layout()
        
        plt.savefig(f"figures/decoding/ml_CLASS_PR_Catboost_lowf_balanced.pdf")
    return per_