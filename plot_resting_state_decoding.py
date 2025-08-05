import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from decoder import compute_ml

df_res = pd.read_csv("results_ml.csv")

features_best = []
best_features = []
df_res_cp = df_res.copy()
df_res_cp["feature"] = df_res_cp["feature"] + "_" + df_res_cp["model"]
locs = df_res_cp["loc"].unique()

for loc in df_res["loc"].unique():
    best_ = df_res_cp.query("loc == @loc ").groupby("feature")["per"].mean().sort_values(ascending=False).head(1).index[0]
    features_best.append(df_res_cp.query("loc == @loc and feature == @best_"))
    best_features.append(best_)
df_best = pd.concat(features_best)

# plot a boxplot with SC_L_ best model 
df_best_C_R = df_best.query("loc == 'C_R_1_'")
# add the linear theta model
df_best_C_R_comp = pd.concat([df_best_C_R, df_res.query("loc == 'C_R_1_' and feature == 'theta' and model == 'Linear'")])

mean_metric_feature_model = df_res.query("loc == 'C_R_1_'").groupby(["feature", "model"])["per"].mean().reset_index()
heatmap_data = mean_metric_feature_model.pivot(index="model", columns="feature", values="per")
# clip data at 0, max
heatmap_data = heatmap_data.clip(lower=0, upper=1)

df_features = pd.read_csv("features_prep_combined_wide.csv")
col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"

#_, pred_best_C_2, true_best_C_2 = compute_ml(df_features, col_score, "burst_amplitude", "NeuralNet", "C_R_1_", return_pred=True)  # Example call to test the function
_, pred_best_C_2, true_best_C_2 = compute_ml(df_features, col_score, "fooof", "Linear", "C_R_1_", return_pred=True)  # Example call to test the function


fig, axes = plt.subplots(1, 4, figsize=(15,6), gridspec_kw={'width_ratios': [0.6, 1, 1, 1]})

# --- Subplot 1 ---
ax = axes[0]
order_model = ["Linear", "NeuralNet"]
sns.boxplot(x="model", y="per", data=df_best_C_R_comp, showmeans=True, showfliers=False,
            boxprops=dict(alpha=0.3), order=order_model, ax=ax)
sns.swarmplot(x="model", y="per", data=df_best_C_R_comp, hue="feature",
              palette="tab10", dodge=False, order=order_model, ax=ax)
for sub in df_best_C_R_comp["subject"].unique():
    sub_data = df_best_C_R_comp.query("subject == @sub")
    ax.plot(sub_data["model"], sub_data["per"], marker='o', label=sub, alpha=0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Best")
# turn upper and right spines off
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# --- Subplot 2 ---

ax = axes[1]
ax.plot(stats.zscore(pred_best_C_2[2]), label="Predicted", color="darkblue")
ax.plot(stats.zscore(true_best_C_2[2]), label="True", color="black")
plt.xlabel("SUDS [a.u.]")
plt.ylabel("Time [a.u.]")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# show the legend
ax.legend(loc='upper right')

ax = axes[2]
sns.boxplot(x="loc", y="per", data=df_best, order=locs,
            showmeans=True, showfliers=False, boxprops=dict(alpha=0.3), ax=ax)
sns.swarmplot(x="loc", y="per", data=df_best, order=locs,
              hue="subject", palette="tab10", dodge=False, ax=ax)
ax.set_xticks(np.arange(len(locs)))
ax.set_xticklabels([f"{loc}\n{best}" for loc, best in zip(locs, best_features)], rotation=90)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# --- Subplot 3 ---
ax = axes[3]
order_models = ["XGB", "RF", "NeuralNet", "SVR_rbf", "SVR_linear", "Linear"]
order_features = ["fft_only", "fft_psd", "Hjorth", "Sharpwave", "fooof", "burst_amplitude", "burst_duration", "burst", "all", "alpha", "beta", "delta", "gamma", "theta"]
heatmap_data = heatmap_data.reindex(index=order_models, columns=order_features)
sns.heatmap(heatmap_data, annot=False, fmt=".2f", center=0, cmap="Blues", ax=ax)
ax.set_title("Mean Rho for Each Feature-Model Combination")

plt.tight_layout()