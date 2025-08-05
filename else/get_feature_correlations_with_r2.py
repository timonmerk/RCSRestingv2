from turtle import pos
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.utils.multiclass import unique_labels

from sklearn import metrics, linear_model
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm
import numpy as np

modalities = ["delta_lfp_left_OvER_interpolate_dayR2", "24h_left_normed_and_delta_lfp_dayR2", 
              "24h_left_normed", "left", "right", "power", "Hjorth", "Hjorth_activity_left", 
              "Hjorth_activity_right", "Hjorth_mobility_left", "Hjorth_mobility_right", 
              "Hjorth_complexity_left", "Hjorth_complexity_right", "all",
              "power_left", "power_right", "Hjorth_left", "Hjorth_right",
               "12h_left", "24h_left", "12h_right", "24h_right", "normed", "6h_left", "6h_right", "12h_left_normed",
               "12h_right_normed", "24h_right_normed", "6h_left_normed", "6h_right_normed", "power_left_sum", "power_right_sum",]

archs = ["logistic", "xgb"] # 

df_comp_out = []

time_range = 1
df = pd.read_csv(f"else/df_features_rick_{time_range}days_normed_by_sum.csv")
df.drop(columns=["Unnamed: 0", "Pxx_left", "Pxx_right"], inplace=True)
subs_remove = ["B009", "B0014", "B020"]
df = df.query("subject not in @subs_remove")

feature_cols = [c for c in df.columns if c not in ["subject", "response", "date", "delta_lfp_left_OvER_interpolate_dayR2"]]
NORM_BY_PREDBS = True
if NORM_BY_PREDBS:
    for sub in df["subject"].unique():
        df_sub = df.query("subject == @sub")
        pre_dbs_mean = df_sub.query("response == 'Pre-DBS'")[feature_cols].mean()
        df.loc[df["subject"] == sub, feature_cols] -= pre_dbs_mean

cols_ = [f for f in df.columns if f!="subject" and f != "date" and f != "response" and "6h" not in f]
df[cols_].corr()

# plot
plt.figure(figsize=(10, 8))
sns.heatmap(df[cols_].corr(), annot=False, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title(f"Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"else/feature_correlation_heatmap_{time_range}days.pdf")
plt.show()