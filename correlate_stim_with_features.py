import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

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

df = pd.read_csv("df_merge_feature_stim_scores.csv")

# ok, idea now get the correlations for rs1 and rs2;
# for each subject-date pair
subjects = df["subject"].unique()
rows_use = []
for subject in subjects:
    dates = df[df["subject"] == subject]["date"].unique()
    for date in dates:
        df_sub_date = df[(df["subject"] == subject) & (df["date"] == date)]
        row_select = None
        if "resting-state1" in df_sub_date["rs_name"].values:
            row_select = df_sub_date[df_sub_date["rs_name"] == "resting-state1"]
        elif "resting-state2" in df_sub_date["rs_name"].values and "resting-state1" not in df_sub_date["rs_name"].values:
            row_select = df_sub_date[df_sub_date["rs_name"] == "resting-state2"]
        if row_select is not None:
            if row_select["Amplitude_mA_left"].iloc[0] != 0 and row_select["Amplitude_mA_right"].iloc[0] != 0:
                rows_use.append(row_select)
df_sel = pd.concat(rows_use).reset_index(drop=True)
df_sel["TEED_left"] = df_sel["Amplitude_mA_left"] * df_sel["PulseWidth_us_left"] * df_sel["Frequency_Hz_left"] / 1000
df_sel["TEED_right"] = df_sel["Amplitude_mA_right"] * df_sel["PulseWidth_us_right"] * df_sel["Frequency_Hz_right"] / 1000


df_sel["TEED_mean"] = (df_sel["TEED_left"] + df_sel["TEED_right"]) / 2
# plot box YBOCS and TEED mean for each patient separatly (left y axis and right y axis), just as plt.plot style (timeline)
pdf_ = PdfPages("TEED_YBOCS_per_patient.pdf")
for patient in df_sel["subject"].unique():
    df_patient = df_sel[df_sel["subject"] == patient]
    df_patient = df_patient.sort_values(by="date")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(df_patient["date"], df_patient["YBOCS II Total Score"], color="blue", marker="o", label="YBOCS II Total Score")
    ax2.plot(df_patient["date"], df_patient["TEED_mean"], color="red", marker="o", label="TEED Mean")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("YBOCS II Total Score", color="blue")
    ax2.set_ylabel("TEED Mean (mJ)", color="red")
    plt.title(f"Patient {patient}: YBOCS II Total Score and TEED Mean over Time")
    fig.tight_layout()
    pdf_.savefig(fig)
    plt.close(fig)
pdf_.close()
    

features_correlate = [c for c in df_sel.columns if (c.startswith("SC_") or c.startswith("C_")) and "psd" not in c]

corr_vecs = []
corr_vecs_ybobs = []
col_score = "YBOCS II Total Score" 

for patient in df_sel["subject"].unique():
    df_patient = df_sel[df_sel["subject"] == patient]
    for hem in ["left", "right"]:
        if hem == "left":
            features_hem = [f for f in features_correlate if f"_L_" in f]
        else:
            features_hem = [f for f in features_correlate if f"_R_" in f]

        
        corr_vector = df_patient[features_hem + ["TEED_" + hem]].corr()["TEED_" + hem]
        corr_vector["subject"] = patient
        corr_vector["Hemisphere"] = hem
        # rename TEED_left to TEED and TEED_right to TEED
        corr_vector = corr_vector.rename({"TEED_" + hem: "TEED"})

        corr_ybocs = df_patient[features_hem + [col_score]].corr()[col_score]
        corr_ybocs["subject"] = patient
        corr_ybocs["Hemisphere"] = hem
        corr_vecs_ybobs.append(corr_ybocs)
        corr_vecs.append(corr_vector)

corr_mat = pd.concat(corr_vecs, axis=1).T.reset_index()
corr_ybocs_mat = pd.concat(corr_vecs_ybobs, axis=1).T.reset_index()
corr_ybocs_l = corr_ybocs_mat[corr_ybocs_mat["Hemisphere"] == "left"]
corr_ybocs_r = corr_ybocs_mat[corr_ybocs_mat["Hemisphere"] == "right"]

corr_mat_L = corr_mat[corr_mat["Hemisphere"] == "left"]
# remove columns with all NaN values
corr_mat_L = corr_mat_L.dropna(axis=1, how='all')
corr_mat_R = corr_mat[corr_mat["Hemisphere"] == "right"]
corr_mat_R = corr_mat_R.dropna(axis=1, how='all')
# those are the correlations for each patient 


col_plt_L = [c for c in corr_mat_L.columns if c.startswith("SC_")]
col_plt_R = [c for c in corr_mat_R.columns if c.startswith("SC_")]
df_plt_L = corr_mat_L[col_plt_L]
df_plt_R = corr_mat_R[col_plt_R]
df_plt_L_ybocs = corr_ybocs_l[col_plt_L]
df_plt_R_ybocs = corr_ybocs_r[col_plt_R]

cols_ = [f[5:] for f in df_plt_L.columns]
df_plt_L.columns = cols_
df_plt_L = df_plt_L[feature_names]
df_plt_L_ybocs.columns = cols_
df_plt_L_ybocs = df_plt_L_ybocs[feature_names]
df_plt_R_ybocs.columns = cols_
df_plt_R_ybocs = df_plt_R_ybocs[feature_names]

cols_ = [f[5:] for f in df_plt_R.columns]
df_plt_R.columns = cols_
df_plt_R = df_plt_R[feature_names]
df_plt_R["sub"]= corr_mat_R["subject"].values
df_plt_L["sub"] = corr_mat_L["subject"].values
df_plt_R["hem"] = "right"
df_plt_L["hem"] = "left"
df_plt_R_ybocs["sub"]= corr_ybocs_r["subject"].values
df_plt_L_ybocs["sub"] = corr_ybocs_l["subject"].values
df_plt_R_ybocs["hem"] = "right"
df_plt_L_ybocs["hem"] = "left"
    
df_comb = pd.concat([df_plt_L, df_plt_R], axis=0).reset_index(drop=True)
df_comb_ybocs = pd.concat([df_plt_L_ybocs, df_plt_R_ybocs], axis=0).reset_index(drop=True)
df_comb_ybocs["type"] = "YBOCS"
df_comb["type"] = "TEED"
df_all = pd.concat([df_comb, df_comb_ybocs], axis=0).reset_index(drop=True)
# pivot s.t. feature_names are all in a column 'feature_name', keep sub, hem, type
df_all_pivot = pd.melt(df_all, id_vars=["sub", "hem", "type"], var_name="feature_name", value_name="correlation")
# ok, but there should be a correlation column for TEED and YBOCS
df_all_pivot_wide = df_all_pivot.pivot_table(index=["sub", "hem", "feature_name"], columns="type", values="correlation").reset_index()
# change to float if possible
df_all_pivot_wide["TEED"] = pd.to_numeric(df_all_pivot_wide["TEED"], errors="coerce")
df_all_pivot_wide["YBOCS"] = pd.to_numeric(df_all_pivot_wide["YBOCS"], errors="coerce")

import matplotlib.pyplot as plt
import seaborn as sns

g = sns.lmplot(
    data=df_all_pivot_wide,
    x="TEED",
    y="YBOCS",
    hue="sub",      # one line per patient
    col="hem",      # left vs right hemispheres
    height=5,
    aspect=1,
    ci=None
)

g.set_titles("{col_name} hemisphere")

# ok now, compare per patient the correlations in a boxplot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("left")
sns.boxplot(data=df_all_pivot.query("hem == 'left'"), x="sub", y="correlation", hue="type",showmeans=True)
sns.swarmplot(data=df_all_pivot.query("hem == 'left'"), x="sub", y="correlation", hue="type", dodge=True, color=".25",)
plt.xticks(rotation=90)
plt.subplot(1, 2, 2)
plt.title("right")
sns.boxplot(data=df_all_pivot.query("hem == 'right'"), x="sub", y="correlation", hue="type", showmeans=True)
sns.swarmplot(data=df_all_pivot.query("hem == 'right'"), x="sub", y="correlation", hue="type", dodge=True, color=".25",)
plt.xlabel("Patient")
plt.xticks(rotation=90)
plt.suptitle("Raw correlation values")
plt.tight_layout()
df_all_pivot["corr_abs"] = df_all_pivot["correlation"].abs()
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("left")
sns.boxplot(data=df_all_pivot.query("hem == 'left'"), x="sub", y="corr_abs", hue="type",showmeans=True)
sns.swarmplot(data=df_all_pivot.query("hem == 'left'"), x="sub", y="corr_abs", hue="type", dodge=True, color=".25",)
plt.xticks(rotation=90)
plt.subplot(1, 2, 2)
sns.boxplot(data=df_all_pivot.query("hem == 'right'"), x="sub", y="corr_abs", hue="type", showmeans=True)
sns.swarmplot(data=df_all_pivot.query("hem == 'right'"), x="sub", y="corr_abs", hue="type", dodge=True, color=".25",)
plt.xlabel("Patient")
plt.title("right")
plt.xticks(rotation=90)
plt.suptitle("Absolute correlation values")
plt.tight_layout()

#

# make a subplot for each patient; left and right should be in one subplot, should be an imgplot with cmap coolwarm, x axis should be left and right hemisphere
patients = df_comb["sub"].unique()
plt.figure(figsize=(18, 6))
for idx, patient in enumerate(patients):
    plt.subplot(1, len(patients), idx + 1)
    df_patient = df_comb[df_comb["sub"] == patient]
    df_patient = df_patient.set_index("hem")
    # remove "sub" from index
    df_patient = df_patient.drop(columns=["sub"])
    df_patient = df_patient.apply(pd.to_numeric, errors="coerce")
    sns.heatmap(df_patient.T, annot=True, cmap="coolwarm", center=0, cbar_kws={"shrink": 0.5}, vmin=-0.5, vmax=0.5)
    plt.title(f'Patient {patient}')
    if idx != 0:
        plt.yticks([], [])
        # remove colorbar
    if idx != len(patients) - 1:
        plt.gca().collections[0].colorbar.remove()
    
    # flip y axis
    plt.gca().invert_yaxis()
plt.suptitle("TEED correlations")
plt.tight_layout()

plt.figure(figsize=(18, 6))
for idx, patient in enumerate(patients):
    plt.subplot(1, len(patients), idx + 1)
    df_patient = df_comb_ybocs[df_comb_ybocs["sub"] == patient]
    df_patient = df_patient.set_index("hem")
    # remove "sub" from index
    df_patient = df_patient.drop(columns=["sub"])
    df_patient = df_patient.apply(pd.to_numeric, errors="coerce")
    sns.heatmap(df_patient.T, annot=True, cmap="coolwarm", center=0, cbar_kws={"shrink": 0.5}, vmin=-0.5, vmax=0.5)
    plt.title(f'Patient {patient}')
    if idx != 0:
        plt.yticks([], [])
    
    # flip y axis
    plt.gca().invert_yaxis()
plt.suptitle("YBOCS")

# ok, question now, are these correlations stronger than YBOCS correlations?


# sns.regplot(data=df_sel, x="Amplitude_mA_left", y="Amplitude_mA_right")

PLOT_STIM_PARAMS = True
if PLOT_STIM_PARAMS:
    df_plt = df.query("rs_name == 'resting-state1'")["Amplitude_mA_left"]
    df_rs1 = df[df["rs_name"] == 'resting-state1']
    df_rs2 = df[df["rs_name"] == 'resting-state2']
    subs = df["subject"].unique()
    df["date"] = pd.to_datetime(df["date"])
    # plot for each subject the Amplitude_mA_left over date, left and right hemisphere in separate subplot
    # make for each patient a different subplot

    # remove all columns that start with C or SC or Unknown
    df_share = df.loc[:, ~df.columns.str.startswith(('C', 'SC', 'Unknown', "Misc", "Unnamed"))]
    df_share.to_csv("df_settings_dates_merged_share.csv")

    with PdfPages("stim_amplitude_over_time.pdf") as pdf:
        for metric in ["Amplitude_mA", "PulseWidth_us", "Frequency_Hz"]:
            missing_all = 0
            all_num_vals = 0
            for patient in subs:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f'Patient {patient}', fontsize=16)
                
                for idx, hem in enumerate(['left', 'right']):
                    ax = axes[idx]
                    patient_data = df[df["subject"] == patient]
                    # sort by date
                    patient_data = patient_data.sort_values(by='date')
                    patient_data_rs1 = patient_data.query("rs_name == 'resting-state1'")
                    patient_data_rs2 = patient_data.query("rs_name == 'resting-state2'")
                    metric_name = f"{metric}_{hem}"
                    
                    # remove NaN values
                    dates_rs1 = patient_data_rs1['date'][patient_data_rs1[metric_name].notna()]
                    values_rs1 = patient_data_rs1[metric_name][patient_data_rs1[metric_name].notna()]
                    ax.plot(dates_rs1, values_rs1, marker='o', label="rs1")

                    dates_rs2 = patient_data_rs2['date'][patient_data_rs2[metric_name].notna()]
                    values_rs2 = patient_data_rs2[metric_name][patient_data_rs2[metric_name].notna()]
                    ax.plot(dates_rs2, values_rs2, marker='x', label="rs2")
                    missing_vals = np.abs(values_rs1.shape[0] - values_rs2.shape[0])
                    all_vals = max(values_rs1.shape[0], values_rs2.shape[0])
                    print(f"Patient {patient}, {metric_name}: Missing values = {missing_vals} out of {all_vals}")
                    missing_all += missing_vals
                    all_num_vals += all_vals
                    # print which one is missing
                    for date in pd.concat([dates_rs1, dates_rs2]).unique():
                        in_rs1 = date in dates_rs1.values
                        in_rs2 = date in dates_rs2.values
                        if in_rs1 and not in_rs2:
                            print(f"  Missing rs2 for date {date.date()}")
                        if in_rs2 and not in_rs1:
                            print(f"  Missing rs1 for date {date.date()}")

                    ax.set_xlabel('Session Date')
                    # Rotate date labels for better readability
                    ax.tick_params(axis='x', rotation=90)
                    # show only months datelabel year-month
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                    # show only every 6 months
                    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=6))
                    ax.set_ylabel(metric)
                    ax.set_title(f'{metric}')
                    ax.legend()
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            print(f"Total missing values for {metric}: {missing_all} out of {all_num_vals}")

    print("PDF saved as stim_amplitude_over_time.pdf")

