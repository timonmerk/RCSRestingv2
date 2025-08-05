from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import pickle
import numpy as np

df_res_rick = pd.read_csv("else/patient_model_results.csv")
df_res_rick = df_res_rick[["pt_id", "TPR", "TNR"]]
# rename pt_id to subject, TPR to tpr, TNR to tnr
df_res_rick.rename(columns={"pt_id": "subject", "TPR": "tpr", "TNR": "tnr"}, inplace=True)
# remove OVERALL and WEIGHTED OVERALL
df_res_rick = df_res_rick.query("subject != 'OVERALL' and subject != 'WEIGHTED OVERALL'")

df_res = pd.read_csv("else/df_comp_ml_LRXGB.csv")

df_best = df_res.groupby("arch_mod_days")[["tnr", "tpr"]].mean()
df_best["mean_tnr_tpr"] = (df_best["tnr"] + df_best["tpr"]) / 2

# get max mean_tnr_tpr
list_best = df_best["mean_tnr_tpr"].sort_values(ascending=False).keys().tolist()
best_arch_mod_days = df_best["mean_tnr_tpr"].idxmax()

cols_ = [f'logistic_delta_lfp_left_OvER_interpolate_dayR2_{i}' for i in [1, 5, 10, 15]]
cols_ += [f'xgb_delta_lfp_left_OvER_interpolate_dayR2_{i}' for i in [1, 5, 10, 15]]

df_plt_ = df_res.query("arch_mod_days in @cols_")
# I want to pivot the table so that metric is tpr and tnr in a single column
df_plt_pivot = df_plt_.melt(id_vars=["arch_mod_days", "num_days", "subject"], value_vars=["tpr", "tnr"], var_name="metric", value_name="value")

df_rick_pivot = df_res_rick.melt(id_vars=["subject"], value_vars=["tpr", "tnr"], var_name="metric", value_name="value")
df_rick_pivot["arch_mod_days"] = "Previous Results"
df_rick_pivot["num_days"] = 2


df_plt_pivot = pd.concat([df_plt_pivot, df_rick_pivot], ignore_index=True)

# [k for k in list(df_best["mean_tnr_tpr"].sort_values(ascending=False).keys()) if "R2" not in k]

PLOT_RES_PREVIOUS_COMPARISON = True
if PLOT_RES_PREVIOUS_COMPARISON:
    cols_sel = ['Previous Results', 'logistic_delta_lfp_left_OvER_interpolate_dayR2_15', 'xgb_delta_lfp_left_OvER_interpolate_dayR2_15',]
    df_plt = df_plt_pivot.query("arch_mod_days in @cols_sel")
    plt.figure(figsize=(4, 6))
    sns.boxplot(data=df_plt, hue="arch_mod_days", y="value", x="metric",
                palette="viridis", boxprops=dict(alpha=0.5), showmeans=True, showfliers=False) # order=order_,
    sns.swarmplot(data=df_plt, hue="arch_mod_days", y="value", x="metric", palette="viridis", dodge=True, alpha=0.5, legend=False) #  order=order_,
    plt.title("TPR and TNR by Architecture and Days")
    plt.xlabel("Architecture and Days")
    plt.ylabel("Value")
    plt.legend(title="Metric")
    plt.savefig("else/plot_tpr_tnr_previous_results.pdf")

# 'logistic_Hjorth_activity_left_15', 
#df_plt_comp = df_res.query("arch_mod_days in @cols_best_fft")
#df_plt_comp_pivot = df_res.melt(id_vars=["arch_mod_days", "num_days"], value_vars=["tpr", "tnr"], var_name="metric", value_name="value")

#df_plt_pivot = pd.concat([df_plt_pivot, df_plt_comp_pivot], ignore_index=True)

# order_.append("Previous Results")
# xtick_labels.append("Previous Results")
# order_ = df_plt_pivot.groupby("arch_mod_days")["value"].mean().sort_values(ascending=False).index.tolist()[::-1]
# xtick_labels = [f"R2_{mod.split('_')[-1]}" for mod in order_]



df_plt_pivot_sel = df_plt_pivot.query("num_days in [1, 5, 10, 15]")
df_plt_pivot_sel["model_type"] = df_plt_pivot_sel["arch_mod_days"].apply(lambda k: "xgb" if "xgb" in k else "logistic")
df_plt_pivot_sel["use"] = df_plt_pivot_sel["arch_mod_days"].apply(lambda k: "R2" if "R2" in k and "delta_lfp" in k and "24h" not in k else "Other")
df_plt_pivot_sel = df_plt_pivot_sel.query("use == 'R2'")

plt.figure(figsize=(5, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=df_plt_pivot_sel.query("metric == 'tnr'"), x="model_type", y="value", hue="num_days",
            palette="viridis", boxprops=dict(alpha=0.5), showmeans=True, showfliers=False)
sns.swarmplot(data=df_plt_pivot_sel.query("metric == 'tnr'"), x="model_type", y="value", hue="num_days",
              dodge=True, alpha=0.5, legend=False, color="gray")
plt.ylabel("TNR")
plt.legend(title="Days")
# upper and right spines are removed
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplot(1, 2, 2)
sns.boxplot(data=df_plt_pivot_sel.query("metric == 'tpr'"), x="model_type", y="value", hue="num_days",
            palette="viridis", boxprops=dict(alpha=0.5), showmeans=True, showfliers=False, legend=False)
sns.swarmplot(data=df_plt_pivot_sel.query("metric == 'tpr'"), x="model_type", y="value", hue="num_days",
              dodge=True, alpha=0.5, legend=False, color="gray")
plt.ylabel("TPR")
plt.suptitle("TPR and TNR by Model Type and Days")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("else/plot_tpr_tnr_by_model_type_and_days.pdf")
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=df_plt_pivot, x="arch_mod_days", y="value", hue="metric",
            palette="viridis", boxprops=dict(alpha=0.5), showmeans=True, showfliers=False) # order=order_,
sns.swarmplot(data=df_plt_pivot, x="arch_mod_days", y="value", hue="metric", palette="viridis", dodge=True, alpha=0.5, legend=False) #  order=order_,
plt.title("TPR and TNR by Architecture and Days")
plt.xlabel("Architecture and Days")
plt.ylabel("Value")
plt.legend(title="Metric")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

df_res.query("arch_mod_days == 'logistic_delta_lfp_left_OvER_interpolate_dayR2_15'")
df_res.query("arch_mod_days == @best_arch_mod_days")

df_comp_out = df_plt_pivot.pivot_table(
    index=["arch_mod_days", "num_days"],  # or add "subject" if needed
    columns="metric",
    values="value",
    aggfunc="mean"  # or "first", "max", etc. depending on your data
).reset_index()[["arch_mod_days", "tnr", "tpr"]]
df_comp_out.query("arch_mod_days == 'Previous Results' or arch_mod_days == @best_arch_mod_days or arch_mod_days == 'logistic_delta_lfp_left_OvER_interpolate_dayR2_15'")