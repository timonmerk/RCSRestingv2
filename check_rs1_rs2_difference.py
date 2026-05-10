import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats

feature_names = [
                 "delta",
                 "theta",
                 "alpha",
                 "beta",
                 "gamma",
                 "offset",
                 "exponent",
                 "raw",
                 "RawHjorth_Activity",
                 "RawHjorth_Complexity",
                 "RawHjorth_Mobility",
                 "Sharpwave_Max_prominence_range_1_12",
                 "Sharpwave_Max_prominence_range_1_5",
                 "Sharpwave_Max_sharpness_range_1_12",
                 "Sharpwave_Max_sharpness_range_1_5",
                 "Sharpwave_Mean_interval_range_1_12",
                 "Sharpwave_Mean_interval_range_1_5",
                 "burst_amplitude_delta",
                 "burst_amplitude_theta",
                 "burst_amplitude_alpha",
                 "burst_amplitude_beta",
                 "burst_amplitude_gamma",
                 "burst_duration_alpha_ms",
                 "burst_duration_beta_ms",
                 "burst_duration_delta_ms",
                 "burst_duration_theta_ms",
                 "burst_duration_gamma_ms",
                 ]

SC_L_features = [f"SC_L_{feat}" for feat in feature_names]
SC_R_features = [f"SC_R_{feat}" for feat in feature_names]

df = pd.read_csv("df_merge_feature_stim_scores.csv")

df["TEED_left"] = df["Amplitude_mA_left"] * (df["Frequency_Hz_left"]) * (df["PulseWidth_us_left"] * 0.001)
df["TEED_right"] = df["Amplitude_mA_right"] * (df["Frequency_Hz_right"]) * (df["PulseWidth_us_right"]* 0.001)
df["TEED_mean"] = df[["TEED_left", "TEED_right"]].mean(axis=1)

patients = df["subject"].unique()
prev_rs2_L = None
prev_rs2_R = None
TEED_L_prev = None
TEED_R_prev = None

all = []

for patient in patients:
    df_sub = df[df["subject"] == patient]
    days = df_sub["date"].unique()
    if patient == "aDBS012":
        print("check")  # aDBS012 had always the same parameters for rs1 and rs2, do TEED diff always 0

    for day in days:
        df_sub_day = df_sub[df_sub["date"] == day]
        rs_names = df_sub_day["rs_name"].unique()
        if "resting-state1" not in rs_names or "resting-state2" not in rs_names:
            continue
        df_rs1 = df_sub_day[df_sub_day["rs_name"] == "resting-state1"]
        df_rs2 = df_sub_day[df_sub_day["rs_name"] == "resting-state2"]

        df_rs1_features_L = df_rs1[SC_L_features].reset_index(drop=True)
        df_rs2_features_L = df_rs2[SC_L_features].reset_index(drop=True)
        df_rs1_features_R = df_rs1[SC_R_features].reset_index(drop=True)
        df_rs2_features_R = df_rs2[SC_R_features].reset_index(drop=True)

        df_rs1_features_L.columns = feature_names
        df_rs2_features_L.columns = feature_names
        df_rs1_features_R.columns = feature_names
        df_rs2_features_R.columns = feature_names

        df_rs1_features_L["hem"] = "left"
        df_rs2_features_L["hem"] = "left"
        df_rs1_features_R["hem"] = "right"
        df_rs2_features_R["hem"] = "right"

        df_rs1_features_L["rs"] = "resting-state1"
        df_rs1_features_R["rs"] = "resting-state1"
        df_rs2_features_L["rs"] = "resting-state2"
        df_rs2_features_R["rs"] = "resting-state2"
        df_add = pd.concat([df_rs1_features_L, df_rs2_features_L, df_rs1_features_R, df_rs2_features_R], ignore_index=True)
        df_add["sub"] = patient
        df_add["day"] = day
        all.append(df_add)
        

        
df_diff = pd.concat(all, ignore_index=True)

df_melt = df_diff.melt(id_vars=["sub", "hem", "rs", "day"], value_vars=feature_names,
                      var_name="feature", value_name="value")

# z-score within subject, hem, feature
df_melt["value_z"] = df_melt.groupby(["sub", "hem", "feature"])["value"].transform(lambda x: (x - x.mean()) / x.std())
# groupby rs 
df_grouped = df_melt.groupby(["sub", "hem", "rs", "feature"])["value_z"].mean().reset_index()

stat_ = stats.permutation_test((df_grouped[df_grouped["rs"] == "resting-state1"]["value_z"],
                                          df_grouped[df_grouped["rs"] == "resting-state2"]["value_z"]),
                                          statistic=lambda x, y: np.mean(x) - np.mean(y),
                                          alternative='two-sided',
                                          n_resamples=5000,
                                          random_state=42)
plt.figure(figsize=(2, 4))
sns.boxplot(data=df_grouped, x="rs", y="value_z", showmeans=True, showfliers=False)
plt.savefig("figures/overall_rs1_rs2_difference_all_features.pdf")



plt.figure(figsize=(20, 15))

for i, feature in enumerate(feature_names):
    ax = plt.subplot(5, 6, i+1)

    df_l_rs1 = df_diff.query("hem == 'left' and rs == 'resting-state1'")[feature].values
    df_l_rs2 = df_diff.query("hem == 'left' and rs == 'resting-state2'")[feature].values
    df_r_rs1 = df_diff.query("hem == 'right' and rs == 'resting-state1'")[feature].values
    df_r_rs2 = df_diff.query("hem == 'right' and rs == 'resting-state2'")[feature].values
    nan_l = np.isnan(df_l_rs1) | np.isnan(df_l_rs2)
    nan_r = np.isnan(df_r_rs1) | np.isnan(df_r_rs2)
    df_l_rs1 = df_l_rs1[~nan_l]
    df_l_rs2 = df_l_rs2[~nan_l]
    df_r_rs1 = df_r_rs1[~nan_r]
    df_r_rs2 = df_r_rs2[~nan_r]
    # run permuation test
    stat = stats.permutation_test((df_l_rs1, df_l_rs2),
                                          statistic=lambda x, y: np.mean(x) - np.mean(y),
                                          alternative='two-sided',
                                          n_resamples=5000,
                                          random_state=42)
    p_val_left = stat.pvalue

    stat = stats.permutation_test((df_r_rs1, df_r_rs2),
                                          statistic=lambda x, y: np.mean(x) - np.mean(y),
                                          alternative='two-sided',
                                          n_resamples=5000,
                                          random_state=42)
    p_val_right = stat.pvalue

    p_val_all = stats.permutation_test((np.concatenate([df_l_rs1, df_r_rs1]), np.concatenate([df_l_rs2, df_r_rs2])),
                                          statistic=lambda x, y: np.mean(x) - np.mean(y),
                                          alternative='two-sided',
                                          n_resamples=5000,
                                          random_state=42).pvalue

    sns.boxplot(
        data=df_diff, hue="rs", y=feature,  # df_mean_diffs to show just one dot per subject
        showmeans=True, ax=ax, showfliers=False,
        hue_order=["resting-state1", "resting-state2"],
         # reduce box width
        width=0.5,
    )
    # sns.swarmplot(
    #     data=df_diff, x="hem", y=feature,  # df_mean_diffs
    #     hue="type", dodge=True, color=".25", ax=ax
    # )

    #ax.set_title(f"{feature}\n p_left: {p_val_left:.3f}, p_right: {p_val_right:.3f}", fontsize=8)
    ax.set_title(f"{feature}\n p={p_val_all:.3f}", fontsize=8)
    # Remove legend from EVERY subplot
    if i != 0:
        ax.legend([], [], frameon=False)

    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig("figures/rs1_rs2_difference_per_feature.pdf")
plt.show()


# question now: how much do the neural features change with TEED?

# correlate each feature with TEED_diff
correlation_results = []
for feature in feature_names:
    for sub in df_diff["sub"].unique():
        for hem in ["left", "right"]:
            df_sub = df_diff[(df_diff["sub"] == sub) & (df_diff["hem"] == hem)]
            corr = df_sub[feature].corr(df_sub["TEED_diff"])
            corr_abs = df_sub[feature].corr(df_sub["TEED_diff"], method='pearson')
            correlation_results.append({
                "sub": sub,
                "feature": feature,
                "correlation": corr,
                "corr_abs": abs(corr),
                "hem": hem,
            })

df_correlation = pd.DataFrame(correlation_results)

# show results in imshow plot, x axis patient, y axis feature, color is correlation, use the 
df_comb = df_correlation.pivot(index="feature", columns=["sub", "hem"], values="correlation")
plt.figure(figsize=(12, 8))
sns.heatmap(df_comb, annot=True, cmap="coolwarm", center=0, vmin=-0.5, vmax=0.5)
plt.title("Correlation between feature changes and TEED difference between rs1 and rs2")
plt.xlabel("Patient and Hemisphere")
plt.ylabel("Feature")
plt.tight_layout()