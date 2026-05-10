import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model, model_selection
import xgboost as xgb
from tqdm import tqdm
from joblib import Parallel, delayed

df_merged = pd.read_csv("FAUS_rs/fau_neural_combined.csv")
cols_neural_features = [col for col in df_merged.columns if col.startswith("C_") or col.startswith("SC_")]
# if col starts with FAU_, remove this part of the column name
col_FAU_replace = [col.replace("FAU_", "") for col in df_merged.columns if col.startswith("FAU_")]
df_merged.rename(columns={old: new for old, new in zip([col for col in df_merged.columns if col.startswith("FAU_")], col_FAU_replace)}, inplace=True)

au_cols = [col for col in df_merged.columns if col.startswith("AU_")]
cols_neural_features_decode = [c for c in cols_neural_features if c.startswith("SC_L_") and "fft" in c and "psd" not in c]

#for sub in df_merged["subject"].unique():
def run_sub(sub):
    dec_res = []
    for AU in tqdm(au_cols):
        for model_name in ["XGB", "Linear"]:
            X_train = df_merged.query("subject != @sub")[cols_neural_features_decode]
            X_test = df_merged.query("subject == @sub")[cols_neural_features_decode]
            y_train = df_merged.query("subject != @sub")[AU]
            y_test = df_merged.query("subject == @sub")[AU]

            if model_name == "XGB":
                model = xgb.XGBRegressor()
            elif model_name == "Linear":
                model = linear_model.LinearRegression()

            idx_nan = X_train.isna().any(axis=1)
            X_train = X_train[~idx_nan]
            y_train = y_train[~idx_nan]
            idx_nan = X_test.isna().any(axis=1)
            X_test = X_test[~idx_nan]
            y_test = y_test[~idx_nan]

            model.fit(X_train, y_train)
            pr = model.predict(X_test)

            rho, p_value = stats.pearsonr(pr, y_test)

            dec_res.append({
                "subject": sub,
                "AU": AU,
                "model": model_name,
                "rho": rho,
                "p_value": p_value
            })
    return dec_res

# test run
#run_sub("aDBS004")
dec_res = Parallel(n_jobs=-1)(delayed(run_sub)(sub) for sub in df_merged["subject"].unique())

df_dec_res = pd.concat([pd.DataFrame(res) for res in dec_res if res is not None], ignore_index=True)
df_dec_res.to_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/figure_4_FAU/fau_decoding_from_neural_LOSO_YBOCS.csv")

df_filtered = df_dec_res[df_dec_res["model"].isin(["XGB", "Linear"])]

df_best_per_au = df_filtered.groupby(["AU", "model"])["rho"].mean().reset_index()

# Get the best model (max rho) per AU
idx_max = df_best_per_au.groupby("AU")["rho"].idxmax()
df_top_models = df_best_per_au.loc[idx_max]

# Now filter original data to only keep those AU-model pairs
au_model_pairs = df_top_models[["AU", "model"]]
df_plot = df_filtered.merge(au_model_pairs, on=["AU", "model"])
# sort by mean rho, avg rho per AU
df_plot_mean = df_plot.groupby("AU")["rho"].mean().reset_index()
df_plot_order = df_plot_mean.sort_values(by="rho", ascending=False)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_plot, x="AU", y="rho", showfliers=False, order=df_plot_order["AU"],
            boxprops=dict(alpha=0.3), showmeans=True)
sns.swarmplot(data=df_plot, x="AU", y="rho", hue="subject", palette="tab10",
              dodge=False, alpha=0.7, order=df_plot_order["AU"])
plt.xticks(rotation=90)
plt.title("Decoding Results for FAU Features")
plt.tight_layout()
plt.savefig("figures/fau_decoding_from_neural_LOSO.pdf")