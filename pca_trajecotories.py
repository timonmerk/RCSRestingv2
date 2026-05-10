import numpy as np
import pandas as pd
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_ = pd.read_csv("features_prep_combined_wide.csv")
col_ = "YBOCS II Total Score"

feature_names = [
                 "delta",
                 "theta",
                 "alpha",
                 "beta",
                 "gamma",
                 "offset",
                 "exponent",
                 ]
feature_names_L = [f"SC_L_{f}" for f in feature_names]
feature_names_R = [f"SC_R_{f}" for f in feature_names]
feature_names_all = feature_names_L + feature_names_R

# run for each patient pca and plot the trajectories
subs_ = X_["subject"].unique()
# normalize features within patients
pca_trajectories = []
y_labels = []
for sub in subs_:
    X_sub = X_[X_["subject"] == sub]
    X = X_[[c for c in X_.columns if c.startswith("SC_") or c == col_]]
    # remove col_
    y = X_sub[col_].values
    X_sub = X_sub.drop(columns=[col_])
    #X_sub = X_sub.dropna(axis=1, how='any')
    y_nan_idx = np.isnan(y)
    X_sub = X_sub[~y_nan_idx]
    y = y[~y_nan_idx]
    # ind where both SC_L_RawHjorth_Activity and SC_R_RawHjorth_Activity are not NaN
    ind_not_nan = ~np.isnan(X_sub["SC_L_RawHjorth_Activity"]) & ~np.isnan(X_sub["SC_R_RawHjorth_Activity"])
    X_sub = X_sub[ind_not_nan]
    y = y[ind_not_nan]

    X_sub = X_sub.drop(columns=["date", "subject"])
    X_sub = X_sub.dropna(axis=1, how='any')
    # remove "SC_" from column names
    #X_sub.columns = [c[3:] for c in X_sub.columns] 
    # select only feature columns
    X_sub = X_sub[feature_names_all]

    scaler = StandardScaler()
    # drop date and subject columns
    
    X_sub_scaled = scaler.fit_transform(X_sub)
    pca = PCA(n_components=2)
    X_sub_pca = pca.fit_transform(X_sub_scaled)
    pca_trajectories.append(X_sub_pca)
    y_labels.append(y)

plt.figure(figsize=(10, 5))
for sub in range(len(subs_)):
    plt.subplot(2, 4, sub+1)
    # set the color of the points according to y_labels[sub]
    plt.scatter(pca_trajectories[sub][:, 0], pca_trajectories[sub][:, 1], c=y_labels[sub], cmap='viridis')
    # add a colorbar
    plt.colorbar()
    plt.plot(pca_trajectories[sub][:, 0], pca_trajectories[sub][:, 1], marker='', label=subs_[sub],
             linewidth=1)
    #plt.legend()
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(subs_[sub])
plt.tight_layout()
plt.show()

# plot all trajectories in one plot
plt.figure(figsize=(7, 7))
for sub in range(len(subs_)):
    plt.plot(pca_trajectories[sub][:, 0], pca_trajectories[sub][:, 1], marker='', label=subs_[sub],
             linewidth=1, color="gray")
    plt.scatter(pca_trajectories[sub][:, 0], pca_trajectories[sub][:, 1], c=y_labels[sub], cmap='viridis', alpha=0.5)
plt.legend()