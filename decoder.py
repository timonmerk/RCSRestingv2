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
               model_type: str = "XGB",
               loc: str = "SC_L_",
               num_months: int = 3,
               return_pred: bool = False):

    per_ = []
    pred_ = []
    true_ = []
    per_ba = None
    for idx, sub in enumerate(df_features["subject"].unique()):

        X_train = df_features.query("subject != @sub").drop(columns=["subject", col_score, "date"])
        X_test = df_features.query("subject == @sub").drop(columns=["subject", col_score, "date"])

        if loc != "all":
            cols_use_loc = [col for col in X_train.columns if col.startswith(loc) and "corr" not in col]
        else:
            cols_use_loc = [col for col in list(X_train.columns) if "corr" not in col]

        X_train_loc = X_train[cols_use_loc]
        X_test_loc = X_test[cols_use_loc]

        if feature != "all":
            if feature != "fft_only":
                if feature == "theta" or feature == "alpha" or feature == "beta" or feature == "gamma" or feature == "delta":
                    cols_use_feature = [col for col in X_train_loc.columns if feature in col and "fft" not in col and "burst" not in col]
                else:
                    if feature == "fft_psd":
                        cols_use_feature = [col for col in X_train_loc.columns if feature in col and int(col.split("_")[-1]) < 124]
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
            continue
    
        y_test = df_features.query("subject == @sub")[col_score]
        y_train = df_features.query("subject != @sub")[col_score]
    
        
        dates = pd.to_datetime(df_features.query("subject == @sub")["date"])

        y_tr_nan_idx = np.isnan(y_train)
        y_te_nan_idx = np.isnan(y_test)
        X_train_feature = X_train_feature[~y_tr_nan_idx]
        y_train = y_train[~y_tr_nan_idx]
        X_test_feature = X_test_feature[~y_te_nan_idx]
        y_test = y_test[~y_te_nan_idx]
        dates = dates[~y_te_nan_idx]

        if model_type == "CatBoost":
            model = CatBoostRegressor(verbose=0, random_seed=42)
        elif model_type == "XGB":
            model = XGBRegressor()
        elif model_type == "Linear":
            model = linear_model.LinearRegression()
        elif model_type == "NeuralNet":
            model = MLPRegressor(random_state=42, max_iter=1000)
        elif model_type == "SVR_linear":
            model = SVR(kernel='linear', C=1.0, epsilon=0.1)
        elif model_type == "SVR_rbf":
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif model_type == "RF":
            model = RandomForestRegressor()

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

            if y_test.shape[0] == 0:
                #print(f"Skipping {sub} due to empty test set")
                continue
        y_test = y_test.values
        y_train = y_train.values
        model.fit(X_train_feature, y_train)
        y_pred = model.predict(X_test_feature)
        
        corr, p = stats.pearsonr(y_test, y_pred)
        if return_pred:
            pred_.append(y_pred)
            true_.append(y_test)
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
            "per" : per,
            "p_value": p,
            "per_ba_class" : per_ba,
            "subject": sub,
            "model": model_type,
            "feature": feature,
            "res_diff_pred": res_diff_pred,
            "res_diff_true": res_diff_true,
            "diff_first_last_true_pred": diff_first_last_true_pred,
            "loc" : loc,
        })

    if return_pred:
        return per_, pred_, true_
    else:
        return per_