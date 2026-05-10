import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# -----------------------
# Config
# -----------------------
CSV_PATH = "features_prep_combined_wide.csv"
Y_COL = "YBOCS II Total Score"

feature_names = [
    "delta", "theta", "alpha", "beta", "gamma", "offset", "exponent",
]
feature_names_L = [f"SC_L_{f}" for f in feature_names]
feature_names_R = [f"SC_R_{f}" for f in feature_names]
FEATURES_ALL = feature_names_L + feature_names_R

# -----------------------
# Load & prepare
# -----------------------
X_ = pd.read_csv(CSV_PATH)

# Keep only what we need (and preserve subject/date for plotting/sorting)
cols_needed = ["subject", "date", Y_COL] + FEATURES_ALL
missing = [c for c in cols_needed if c not in X_.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = X_[cols_needed].copy()

# Optional: ensure chronological order within subject
# (if date is string, try to parse)
if not np.issubdtype(df["date"].dtype, np.datetime64):
    with pd.option_context('mode.chained_assignment', None):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values(["subject", "date"], kind="mergesort")

# Drop rows with any NaN in Y or features
df = df.dropna(subset=[Y_COL] + FEATURES_ALL).reset_index(drop=True)

# -----------------------
# Single normalization + single PCA on ALL rows
# -----------------------
X = df[FEATURES_ALL].values
y = df[Y_COL].values
subs = df["subject"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, svd_solver="full", random_state=0)
X_pca = pca.fit_transform(X_scaled)

# Add back to dataframe for easy grouping
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# For consistent colorbars across plots
y_min, y_max = float(np.min(y)), float(np.max(y))

# -----------------------
# Per-subject trajectories (same PCA space)
# -----------------------
unique_subs = df["subject"].unique()
n_subs = len(unique_subs)

ncols = 4
nrows = int(np.ceil(n_subs / ncols))
plt.figure(figsize=(4*ncols, 3.5*nrows))

for i, sub in enumerate(unique_subs, start=1):
    dsub = df[df["subject"] == sub]
    ax = plt.subplot(nrows, ncols, i)
    sc = ax.scatter(dsub["PC1"], dsub["PC2"], c=dsub[Y_COL], cmap="viridis",
                    vmin=y_min, vmax=y_max, s=18)
    # connect points in time order
    ax.plot(dsub["PC1"].values, dsub["PC2"].values, linewidth=1, alpha=0.8)
    ax.set_title(str(sub))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

# One shared colorbar
cbar = plt.colorbar(sc, ax=plt.gcf().get_axes(), shrink=0.85)
cbar.set_label(Y_COL)
plt.tight_layout()
plt.show()

# -----------------------
# Combined plot
# -----------------------
plt.figure(figsize=(7, 7))
for sub in unique_subs:
    dsub = df[df["subject"] == sub]
    plt.plot(dsub["PC1"], dsub["PC2"], linewidth=1, alpha=0.6, color="gray")
    plt.scatter(dsub["PC1"], dsub["PC2"], c=dsub[Y_COL], cmap="viridis",
                vmin=y_min, vmax=y_max, s=18, alpha=0.7, label=str(sub))

# only show unique labels once
handles, labels = plt.gca().get_legend_handles_labels()
lab_seen, handles_unique, labels_unique = set(), [], []
for h, lab in zip(handles, labels):
    if lab not in lab_seen:
        lab_seen.add(lab)
        handles_unique.append(h)
        labels_unique.append(lab)

#plt.legend(handles_unique, labels_unique, bbox_to_anchor=(1.02, 1), loc="upper left")
cbar = plt.colorbar()
cbar.set_label(Y_COL)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Patient trajectories in a shared PCA space")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 7))

# build a color map from patient IDs
cmap = cm.get_cmap("tab10", len(unique_subs))  # tab10 or tab20 for distinct colors
color_map = {sub: cmap(i) for i, sub in enumerate(unique_subs)}

for sub in unique_subs:
    dsub = df[df["subject"] == sub]
    plt.plot(dsub["PC1"], dsub["PC2"], linewidth=1, alpha=0.6, color=color_map[sub])
    plt.scatter(dsub["PC1"], dsub["PC2"], color=color_map[sub], s=18, alpha=0.7, label=str(sub))

# legend for patients
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Subject")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Patient trajectories in a shared PCA space (color = subject)")
plt.tight_layout()
plt.show()