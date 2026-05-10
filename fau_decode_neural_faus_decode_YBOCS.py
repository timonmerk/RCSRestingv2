import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
import xgboost as xgb
from scipy import stats
from tqdm import tqdm

df = pd.read_csv("FAUS_rs/fau_neural_combined.csv")
score_col = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
l_res = []
for sub in tqdm(df["subject"].unique()):
    for model_name in ["XGB", "Linear"]:
        for feature_mod in ["neural", "AU", "comb"]:
            X_train = df.query("subject != @sub")
            X_test = df.query("subject == @sub")
            y_train = X_train[score_col]
            y_test = X_test[score_col]

            if feature_mod == "neural":
                col_use = [c for c in X_train.columns if c.startswith("SC_L_") and "fft" in c and "psd" not in c]
                X_train = X_train[col_use]
                X_test = X_test[col_use]
            elif feature_mod == "AU":
                col_use = [c for c in X_train.columns if c.startswith("FAU_AU")]
                col_use = ["FAU_AU_R6"]
                X_train = X_train[col_use]
                X_test = X_test[col_use]
            elif feature_mod == "comb":
                col_use = [c for c in X_train.columns if (c.startswith("SC_L_") and "fft" in c and "psd" not in c) or c == "FAU_AU_R6"]
                X_train = X_train[col_use]
                X_test = X_test[col_use]

            if model_name == "XGB":
                model = xgb.XGBRegressor()
            elif model_name == "Linear":
                model = linear_model.LinearRegression()
            idx_nan = X_train.isna().any(axis=1)
            idx_nan |= y_train.isna()
            X_train = X_train[~idx_nan]
            y_train = y_train[~idx_nan]
            idx_nan = X_test.isna().any(axis=1)
            X_test = X_test[~idx_nan]
            y_test = y_test[~idx_nan]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rho, p = stats.pearsonr(y_test, y_pred)
            l_res.append({
                "subject": sub,
                "model": model_name,
                "features": feature_mod,
                "score": rho,
                "p_value": p
            })

df_res = pd.DataFrame(l_res)

df_res["feature_model"] = df_res["features"] + "_" + df_res["model"]
df_res.groupby("feature_model")["score"].mean().reset_index()
PATH_OUT = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/figure_4_FAU"
df_res.to_csv(os.path.join(PATH_OUT, "combined_fau_and_neural_decoding_results_LOSO_YBOCS.csv"), index=False)
plt.figure(figsize=(5, 6))
sns.boxplot(x="model", y="score", hue="features", data=df_res,
            showmeans=True, showfliers=False, boxprops=dict(alpha=0.3))
sns.swarmplot(x="model", y="score", hue="features", data=df_res,
            palette="tab10", dodge=True, alpha=0.7)
plt.xticks(rotation=90)
plt.title("LOSO")
plt.tight_layout()

