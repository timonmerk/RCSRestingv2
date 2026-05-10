import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from decoder import compute_ml
import pickle
import copy
import random

def permutationTest(x, y, plot_distr=True, x_unit=None, p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distr.
    y (np array) : first distr.
    plot_distr (boolean) : if True: plot permutation histplot and ground truth
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here absolute difference of
    distribution means
    p (float) : p value of permutation test

    """
    # Compute ground truth difference
    gT = np.abs(np.average(x) - np.average(y))

    pV = np.concatenate((x, y), axis=0)
    pS = copy.copy(pV)
    # Initialize permutation:
    pD = []
    # Permutation loop:
    for i in range(0, p):
        # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled
        # distributions and store it in pD:
        pD.append(
            np.abs(
                np.average(pS[0 : int(len(pS) / 2)])
                - np.average(pS[int(len(pS) / 2) :])
            )
        )

    # Calculate p-value
    if gT < 0:
        p_val = len(np.where(pD <= gT)[0]) / p
    else:
        p_val = len(np.where(pD >= gT)[0]) / p

    if plot_distr:
        plt.hist(pD, bins=30, label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth " + x_unit + "=" + str(gT) + " p=" + str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()
    return gT, p_val


df_res = pd.read_csv("results_ml.csv")

# rename subject to sub
df_res = df_res.rename(columns={"subject": "sub"})

per_metric = "ccc"

d_save = {}

features_best = []
corr_best = []
best_features = []
df_res_cp = df_res.copy()
df_res_cp["pca"] = df_res_cp["pca"].map({True: "pca", False: "no_pca"})

df_res_cp["feature"] = df_res_cp["feature"] + "_" + df_res_cp["model"] + "_" + df_res_cp["pca"]
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
df_best_SC_L_comp = df_best.query("loc == @region")
# add the linear theta model
df_best_SC_L_comp = pd.concat([df_best_SC_L_comp, df_res.query("loc == @region and feature == 'theta' and model == 'Linear'")])

mean_metric_feature_model = df_res.query("loc == @region").groupby(["feature", "model"])[per_metric].mean().reset_index()
heatmap_data = mean_metric_feature_model.pivot(index="model", columns="feature", values=per_metric)
# clip data at 0, max
heatmap_data = heatmap_data.clip(lower=0, upper=1)

df_features = pd.read_csv("features_prep_combined_wide.csv")
col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"

model = df_best.query("loc == @region")["model"].iloc[0]
feature = df_best.query("loc == @region")["feature"].iloc[0]
pca_ = df_best.query("loc == 'SC_L_'")["pca"].iloc[0]
feature = feature[:-len(model)-1-len(pca_)-1]
pca_ = False if pca_ == "no_pca" else True


_, pred_best, true_best = compute_ml(df_features, col_score, feature, model, region, pca_, return_pred=True) 
#_, pred_best_C_2, true_best_C_2 = compute_ml(df_features, col_score, "fooof", "Linear", "C_R_1_", return_pred=True)  

subjects = df_best["sub"].unique()

##### DATA comp. L vs R SC
comp_L_R_SC = []
l_VCVS = df_best.query("loc == 'SC_L_'")[per_metric].values
r_VCVS = df_best.query("loc == 'SC_R_'")[per_metric].values
for sub in subjects:
    rho_L = df_best.query("sub == @sub and loc == 'SC_L_'")[per_metric].values
    rho_R = df_best.query("sub == @sub and loc == 'SC_R_'")[per_metric].values
    #if len(rho_L) > 0 and len(rho_R) > 0:
    comp_L_R_SC.append(rho_L[0] - rho_R[0])

_, p_val_comp_L_R_SC = permutationTest(np.array(comp_L_R_SC),
                           np.zeros(len(comp_L_R_SC)),
                           plot_distr=False, x_unit=per_metric, p=5000)

df_plt_comp_SC_LR = df_best.query("loc in ['SC_L_', 'SC_R_']")
d_save["df_plt_comp_SC_LR"] = df_plt_comp_SC_LR
d_save["p_val_comp_L_R_SC"] = p_val_comp_L_R_SC

#### COMP MED vs LAT OFC
# set ofc_lat to 1 if C_L_1_ or C_R_1_ is in loc
df_comp_med_lat = df_best.copy().query("loc in ['C_L_1_', 'C_R_1_', 'C_L_2_', 'C_R_2_']")
df_comp_med_lat["OFC_LAT"] = df_comp_med_lat["loc"].apply(lambda x: "LAT" if "C_L_1_" in x or "C_R_1_" in x else "MED")
# select best model per subject and OFC_LAT
df_best_OFC_LAT = df_comp_med_lat.sort_values(by=per_metric, ascending=False).groupby(["sub", "OFC_LAT"]).head(1)

_, p_val_comp_OFC_LAT = permutationTest(
    df_best_OFC_LAT.query("OFC_LAT == 'MED'")[per_metric].values - df_best_OFC_LAT.query("OFC_LAT == 'LAT'")[per_metric].values,
    np.zeros(len(df_best_OFC_LAT.query("OFC_LAT == 'MED'")[per_metric].values)),
    plot_distr=False, x_unit=per_metric, p=5000
)

d_save["df_best_OFC_LAT"] = df_best_OFC_LAT
d_save["p_val_comp_OFC_LAT"] = p_val_comp_OFC_LAT

# plot comp. ECoG VCVCS
#df_best["VCVS"] = df_best["loc"].apply(lambda x: "VCVS" if "SC" in x elif x == "all" "ECoG")
# set VCVS to ECoG if loc is C_L_1_, C_L_2_, C_R_1_, C_R_2_
# set if it starts with SC to VCVS
# if it's all, set to all
df_best["VCVS"] = df_best["loc"].apply(lambda x: "VCVS" if "SC" in x else ("ECoG" if x in ["C_L_1_", "C_L_2_", "C_R_1_", "C_R_2_"] else "all"))

df_best_VCVS_comp = df_best.sort_values(by=per_metric, ascending=False).groupby(["VCVS", "sub"]).head(1)

subs_comp = [9, 10, 11, 12]
diff_per_VCVS_ECOG = []
diff_per_all_ECOG = []
diff_per_all_VCVS = []

for sub in subs_comp:
    rho_VCVS = df_best_VCVS_comp.query("sub == @sub and VCVS == 'VCVS'")[per_metric].values
    rho_ECoG = df_best_VCVS_comp.query("sub == @sub and VCVS == 'ECoG'")[per_metric].values
    rho_all = df_best_VCVS_comp.query("sub == @sub and VCVS == 'all'")[per_metric].values
    if len(rho_VCVS) > 0 and len(rho_ECoG) > 0:
        diff_per_VCVS_ECOG.append(rho_VCVS[0] - rho_ECoG[0])
        diff_per_all_ECOG.append(rho_all[0] - rho_ECoG[0])
        diff_per_all_VCVS.append(rho_all[0] - rho_VCVS[0])

_, p_val_comp_VCVS_ECOG = permutationTest(np.array(diff_per_VCVS_ECOG),
                           np.zeros(len(diff_per_VCVS_ECOG)),
                           plot_distr=False, x_unit=per_metric, p=5000)
_, p_val_comp_VCVS_ALL = permutationTest(np.array(diff_per_all_VCVS),
                           np.zeros(len(diff_per_all_VCVS)),
                           plot_distr=False, x_unit=per_metric, p=5000)
_, p_val_comp_ALL_ECOG = permutationTest(np.array(diff_per_all_ECOG),
                           np.zeros(len(diff_per_all_ECOG)),
                           plot_distr=False, x_unit=per_metric, p=5000)

d_save["df_best_VCVS_comp"] = df_best_VCVS_comp
d_save["p_val_comp_VCVS_ECOG"] = p_val_comp_VCVS_ECOG
d_save["p_val_comp_VCVS_ALL"] = p_val_comp_VCVS_ALL
d_save["p_val_comp_ALL_ECOG"] = p_val_comp_ALL_ECOG

d_save["heatmap_data"] = heatmap_data
d_save["df_best_SC_L_comp"] = df_best_SC_L_comp
d_save["model"] = model

d_save["per_ex"] = corr_best
d_save["pred_best_VCVS_L"] = pred_best
d_save["true_best_VCVS_L"] = true_best
d_save["df_res"] = df_res


pickle.dump(d_save, open("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/figure_2/figure_2_data_YBOCS.pkl", "wb"))

fig, axes = plt.subplots(1, 4, figsize=(15,6), gridspec_kw={'width_ratios': [0.6, 1, 1, 1]})
# --- Subplot 1 ---
ax = axes[0]
order_model = ["Linear", model]
sns.boxplot(x="model", y=per_metric, data=df_best_SC_L_comp, showmeans=True, showfliers=False,
            boxprops=dict(alpha=1, facecolor='white', edgecolor='black'), order=order_model, ax=ax)
#sns.swarmplot(x="model", y=per_metric, data=df_best_C_R_comp, hue="feature",
#              palette="tab10", ddge=False, order=order_model, ax=ax)
color_sub_viridis = sns.color_palette("viridis", n_colors=len(df_best_SC_L_comp["subject"].unique()))
for i, sub in enumerate(df_best_SC_L_comp["subject"].unique()):
    sub_data = df_best_SC_L_comp.query("subject == @sub")
    ax.plot(sub_data["model"], sub_data[per_metric], marker='o', label=sub, color=color_sub_viridis[i], alpha=0.7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Best")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel(f"{per_metric}")

# --- Subplot 2 ---
ax = axes[1]
ax.plot(pred_best[np.argmax(corr_best)], label="Predicted", color="darkblue")
ax.plot(true_best[np.argmax(corr_best)], label="True", color="black")
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
