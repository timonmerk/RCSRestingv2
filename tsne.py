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
# import tsne
from sklearn.manifold import TSNE

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

df_X = df_features.drop(columns=["subject", "date", "YBOCS II Total Score", "response"])
# run TNSE
tsne = TSNE(n_components=2, perplexity=30)
X_embedded = tsne.fit_transform(df_X)   

plt.figure(figsize=(5, 5))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=df_features["subject"], s=40, alpha=0.8)
plt.title("t-SNE Visualization of Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc='upper left')
# omit spines
sns.despine(trim=True, right=True, top=True)
plt.tight_layout()
plt.savefig("figures/tsne_subjects.pdf")


plt.figure(figsize=(5, 5))
plt.scatter(x=X_embedded[:, 0], y=X_embedded[:, 1], c=df_features["response"], cmap='viridis', s=100)
plt.colorbar(label="Response")
plt.title("t-SNE Visualization of Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
sns.despine(trim=True, right=True, top=True)
plt.savefig("figures/tsne_response.pdf")

plt.figure(figsize=(5, 5))
plt.scatter(x=X_embedded[:, 0], y=X_embedded[:, 1], c=df_features[col_score], cmap='viridis', s=100)
plt.colorbar(label=col_score)
plt.title("t-SNE Visualization of Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
sns.despine(trim=True, right=True, top=True)
plt.savefig("figures/tsne_YBOCS2_total.pdf")