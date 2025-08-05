import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
# i want create a multipage pdf with matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import permutation_test
from datetime import datetime

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


df_corr = pd.read_csv("correlation_features.csv")
df_corr["loc"] = ["Cortex" if "Cortex" in ch else "VCVS" for ch in df_corr["channel"]]
df_corr["hem"] = ["left" if "left" in ch else "right" for ch in df_corr["channel"]]
df_corr["correlation"] = df_corr["correlation"].abs()

df_ = pd.read_csv("features_prep_combined.csv")
df_["date"] = df_["file"].apply(get_date)
df_["date"] = df_["date"].apply(convert_to_datetime)

score_column = "YBOCS II Total Score"
subject = "aDBS012"


features_names = df_["feature_name"].unique()
chs_plt = ["VCVS_left", "VCVS_right", "Cortex_left", "Cortex_right"]
#plt.figure()
df_res = []

# Get top 10 Cortex features
idx_10_best_cortex = df_corr.query(
    "score_column == @score_column and subject == @subject and loc == 'Cortex' and hem == 'left'"
)["correlation"].nlargest(10)

# Get top VCVS feature
idx_10_best_vcvs = df_corr.query(
    "score_column == @score_column and subject == @subject and loc == 'VCVS' and hem == 'left'"
)["correlation"].nlargest(10)

idx_cortex = idx_10_best_cortex.index[0]
feature_best_cortex = df_corr.loc[idx_cortex]["feature_name"]
feature_hem_cortex = df_corr.loc[idx_cortex]["hem"]

idx_vcvs = idx_10_best_vcvs.index[0]
feature_best_vcvs = df_corr.loc[idx_vcvs]["feature_name"]
feature_hem_vcvs = df_corr.loc[idx_vcvs]["hem"]

ch_plt = "Cortex_left"
df_sub_c = df_.query("subject == @subject and feature_name == @feature_best_cortex and new_ch == @ch_plt")
df_f_g_c = df_sub_c.groupby("date")[["feature_value", score_column]].mean()    # FILE

idx_not_na = df_f_g_c.index[df_f_g_c[score_column].notna() & df_f_g_c["feature_value"].notna()]
df_f_g_c = df_f_g_c.loc[idx_not_na]
corr_c, p = stats.pearsonr(df_f_g_c["feature_value"], df_f_g_c[score_column])

ch_plt = "VCVS_left"
df_sub_v = df_.query("subject == @subject and feature_name == @feature_best_vcvs and new_ch == @ch_plt")
df_f_g_v = df_sub_v.groupby("date")[["feature_value", score_column]].mean()    # FILE 

idx_not_na = df_f_g_v.index[df_f_g_v[score_column].notna() & df_f_g_v["feature_value"].notna()]
df_f_g_v = df_f_g_v.loc[idx_not_na]
corr_v, p_v = stats.pearsonr(df_f_g_v["feature_value"], df_f_g_v[score_column])

df_c = pd.DataFrame({
    "feature_value": df_f_g_c["feature_value"],
    score_column: df_f_g_c[score_column],
    "loc": "Cortex",
    "date": df_f_g_c.index
})

if corr_v < 0:
    df_f_g_v["feature_value"] = df_f_g_v["feature_value"] * -1
if corr_c < 0:
    df_f_g_c["feature_value"] = df_f_g_c["feature_value"] * -1

df_c["feature_value"] = stats.zscore(df_c["feature_value"])
df_v = pd.DataFrame({
    "feature_value": df_f_g_v["feature_value"],
    score_column: df_f_g_v[score_column],
    "loc": "VCVS",
    "date": df_f_g_v.index
})
df_v["feature_value"] = stats.zscore(df_v["feature_value"])

df_comb = pd.concat([df_c, df_v]).reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), sharey=False,
                               gridspec_kw={'width_ratios': [1, 2]},)

# === LEFT PLOT: Regression (Cortex vs. VCVS)
sns.regplot(
    data=df_comb[df_comb["loc"] == "VCVS"],
    x="feature_value",
    y=score_column,
    ci=95,
    color=sns.color_palette("viridis")[2],
    ax=ax1,
    label="VCVS"
)
sns.regplot(
    data=df_comb[df_comb["loc"] == "Cortex"],
    x="feature_value",
    y=score_column,
    ci=95,
    color=sns.color_palette("viridis")[4],
    ax=ax1,
    label="Cortex"
)

ax1.set_title(f"{subject}\nC:{feature_best_cortex} / V:{feature_best_vcvs}\n"
              f"rho c: {corr_c:.2f} / rho v: {corr_v:.2f}", fontsize=9)
ax1.set_xlabel("Feature Value (z-scored)")
ax1.set_ylabel(score_column)
ax1.legend(title="Location")

# === RIGHT PLOT: Time-series with twin y-axis
# Left y-axis
sns.lineplot(
    x="date", y="feature_value", hue="loc",
    data=df_comb, palette="viridis", ax=ax2
)
ax2.set_ylabel("Feature Value")
ax2.set_xlabel("Date")
ax2.legend(title="Location", loc="upper left")

# Right y-axis
ax3 = ax2.twinx()
sns.lineplot(
    x="date", y=score_column, data=df_comb,
    color="black", label=score_column, ax=ax3
)
ax3.set_ylabel(score_column)
ax3.legend(loc="upper right")

# Rotate x-tick labels
plt.setp(ax2.get_xticklabels(), rotation=90)

# Final layout
plt.tight_layout()
plt.savefig("figures/examplary_correlation_plot_best_features.pdf")