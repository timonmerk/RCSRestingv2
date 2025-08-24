import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from decoder import compute_ml

df_res = pd.read_csv("results_ml.csv")

per_metric = "ccc"

features_best = []
corr_best = []
best_features = []
df_res_cp = df_res.copy()
df_res_cp["feature"] = df_res_cp["feature"] + "_" + df_res_cp["model"]
locs = df_res_cp["loc"].unique()

for loc in df_res["loc"].unique():
    best_ = df_res_cp.query("loc == @loc ").groupby("feature")[per_metric].mean().sort_values(ascending=False).head(1).index[0]
    features_best.append(df_res_cp.query("loc == @loc and feature == @best_"))
    corr_best.append(df_res_cp.query("loc == @loc and feature == @best_")[per_metric].values[0])
    best_features.append(best_)
df_best = pd.concat(features_best)

# plot a boxplot with SC_L_ best model 
region = df_best.groupby("loc")[per_metric].mean().idxmax()
region = "SC_L_"
df_best_C_R = df_best.query("loc == @region")
# add the linear theta model
df_best_C_R_comp = pd.concat([df_best_C_R, df_res.query("loc == @region and feature == 'theta' and model == 'Linear'")])

mean_metric_feature_model = df_res.query("loc == @region").groupby(["feature", "model"])[per_metric].mean().reset_index()
heatmap_data = mean_metric_feature_model.pivot(index="model", columns="feature", values=per_metric)
# clip data at 0, max
heatmap_data = heatmap_data.clip(lower=0, upper=1)

df_features = pd.read_csv("features_prep_combined_wide.csv")
col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"

model = df_best.query("loc == @region")["model"].iloc[0]
feature = df_best.query("loc == @region")["feature"].iloc[0]
feature = feature[:-len(model)-1]

#_, pred_best_C_2, true_best_C_2 = compute_ml(df_features, col_score, "burst_amplitude", "RF", "SC_R_", return_pred=True)  # Example call to test the function
_, pred_best_C_2, true_best_C_2 = compute_ml(df_features, col_score, feature, model, region, return_pred=True) 
#_, pred_best_C_2, true_best_C_2 = compute_ml(df_features, col_score, "fooof", "Linear", "C_R_1_", return_pred=True)  # Example call to test the function


fig, axes = plt.subplots(1, 4, figsize=(15,6), gridspec_kw={'width_ratios': [0.6, 1, 1, 1]})
# --- Subplot 1 ---
ax = axes[0]
order_model = ["Linear", model]
sns.boxplot(x="model", y=per_metric, data=df_best_C_R_comp, showmeans=True, showfliers=False,
            boxprops=dict(alpha=1, facecolor='white', edgecolor='black'), order=order_model, ax=ax)
#sns.swarmplot(x="model", y=per_metric, data=df_best_C_R_comp, hue="feature",
#              palette="tab10", ddge=False, order=order_model, ax=ax)
color_sub_viridis = sns.color_palette("viridis", n_colors=len(df_best_C_R_comp["subject"].unique()))
for i, sub in enumerate(df_best_C_R_comp["subject"].unique()):
    sub_data = df_best_C_R_comp.query("subject == @sub")
    ax.plot(sub_data["model"], sub_data[per_metric], marker='o', label=sub, color=color_sub_viridis[i], alpha=0.7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Best")
# turn upper and right spines off
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# set ylabel
ax.set_ylabel(f"{per_metric}")

# --- Subplot 2 ---
ax = axes[1]
ax.plot(pred_best_C_2[np.argmax(corr_best)], label="Predicted", color="darkblue")
ax.plot(true_best_C_2[np.argmax(corr_best)], label="True", color="black")
ax.set_ylabel("Y-BOCS II Total Score")
ax.set_xlabel("Time [a.u.]")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title(f"{region} {model} {feature}")
ax.legend(loc='upper right')

ax = axes[2]
sns.boxplot(x="loc", y=per_metric, data=df_best, order=locs,
            showmeans=True, showfliers=False, boxprops=dict(alpha=1, facecolor='white', edgecolor='black', ), ax=ax)
sns.swarmplot(x="loc", y=per_metric, data=df_best, order=locs,
              hue="subject", palette="viridis", dodge=False, ax=ax)
ax.set_xticks(np.arange(len(locs)))
ax.set_xticklabels([f"{loc}\n{best}" for loc, best in zip(locs, best_features)], rotation=90)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# --- Subplot 3 ---
ax = axes[3]
order_models = ["XGB", "RF", "NeuralNet", "SVR_rbf", "SVR_linear", "Linear"]
order_features = ["fft_only", "fft_psd", "Hjorth", "Sharpwave", "fooof", "burst_amplitude", "burst_duration", "burst", "all", "alpha", "beta", "delta", "gamma", "theta"]
heatmap_data = heatmap_data.reindex(index=order_models, columns=order_features)
sns.heatmap(heatmap_data, annot=False, fmt=".2f", cmap="viridis", ax=ax)
ax.set_title("Mean Rho for Each Feature-Model Combination")
plt.tight_layout()
plt.savefig("figures/plot_best_model_performance.pdf")


df_comp_LR_SC = df_best.query("loc == 'SC_L_' or loc == 'SC_R_'")
plt.figure(figsize=(2.5, 5))
# show in a boxplot and swarmplot the ccc values for SC_L_ and SC_R_; and connect ind. subjects with lines
sns.boxplot(x="loc", y=per_metric, data=df_comp_LR_SC, order=["SC_L_", "SC_R_"],
            showmeans=True, showfliers=False, boxprops=dict(alpha=1, facecolor='white', edgecolor='black', ))
sns.swarmplot(x="loc", y=per_metric, data=df_comp_LR_SC, order=["SC_L_", "SC_R_"],
              hue="subject", palette="viridis", dodge=False)
for sub in df_comp_LR_SC["subject"].unique():
    sub_data = df_comp_LR_SC.query("subject == @sub")
    if len(sub_data) == 2:
        ax = plt.gca()
        ax.plot(sub_data["loc"], sub_data[per_metric], label=sub, color="gray", alpha=0.7)
plt.xticks(ticks=[0, 1], labels=["SC_L_", "SC_R_"])
plt.ylabel(f"{per_metric}")
plt.tight_layout()

df_comp_C_SC = df_best.query("loc.str.startswith('C_') or loc.str.startswith('SC_')")
df_comp_C_SC["SC"] = df_comp_C_SC["loc"].str.startswith("SC_")
# groupby SC and get the best per subject
mean_metric_feature_model = df_comp_C_SC.groupby(["subject", "SC"])[per_metric].max().reset_index()

order_ = [True, False]

plt.figure(figsize=(2.5, 5))
sns.boxplot(x="SC", y=per_metric, data=mean_metric_feature_model, order=order_,
            showmeans=True, showfliers=False, boxprops=dict(alpha=1, facecolor='white', edgecolor='black', ))
sns.swarmplot(x="SC", y=per_metric, data=mean_metric_feature_model, order=order_,
              hue="subject", palette="viridis", dodge=False)

for sub in mean_metric_feature_model["subject"].unique():
    sub_data = mean_metric_feature_model.query("subject == @sub")
    if len(sub_data) == 2:
        ax = plt.gca()
        ax.plot(sub_data["SC"].astype(int), sub_data[per_metric][::-1], label=sub, color="gray", alpha=0.7)

plt.xticks(ticks=[0, 1], labels=["SC_", "C_"])
plt.ylabel(f"{per_metric}")
plt.tight_layout()

# ok, L + R makes complete sense
# compare all ALL
