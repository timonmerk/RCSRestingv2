import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


for INCLUDE_BEHAV, INLCUDE_STIM in [(False, True), (True, False), (True, True), (False, False), ]:
    print(f"INCLUDE_BEHAV: {INCLUDE_BEHAV}, INLCUDE_STIM: {INLCUDE_STIM}")

    df_stim_params_final = pd.read_csv("stim params/StimParams_Processed_all_times_2026.csv")

    df_neural = pd.read_csv("/Users/Timon/Documents/Houston/whisper/audio_neural_features_combined_rs.csv")
    df_neural["date"] = pd.to_datetime(df_neural["date"])
    l_behavior = [col for col in df_neural.columns if not col.startswith("C_") and not col.startswith("SC") and not col.startswith("Unknown") and not col.startswith("Misc")]
    df_behavior = df_neural[["subject", "date"] + l_behavior]
    # cut out columns 1 to 42
    df_behavior = df_behavior.drop(df_behavior.columns[4:42], axis=1)
    df_behavior = df_behavior.loc[:, ~df_behavior.columns.duplicated()]
    df_behavior = df_behavior.drop(columns=["sub"])

    df = pd.read_csv("df_merge_feature_stim_scores.csv") 
    df['YBOCS II Total Score']
    df["date"] = pd.to_datetime(df["date"])


    neural_features = [col for col in df.columns if (col.startswith("SC") or col.startswith("C_")) and "psd" not in col]
    # stimulation features are either 
    stim_l = ['Amplitude_mA_left', 'Amplitude_mA_right', 'Frequency_Hz_left', 'Frequency_Hz_right', 'PulseWidth_us_left', 'PulseWidth_us_right', ]

    if INLCUDE_STIM:
        df_mean_neural_score_stim = df.groupby(["subject", "date"])[["YBOCS II Total Score"] + neural_features + stim_l].mean().reset_index()
    else:
        df_mean_neural_score_stim = df.groupby(["subject", "date"])[["YBOCS II Total Score"] + neural_features].mean().reset_index()
    df_mean_neural_score_stim["date"] = pd.to_datetime(df_mean_neural_score_stim["date"])

    # merge with behavior
    if INCLUDE_BEHAV:
        df_merge = pd.merge(df_mean_neural_score_stim, df_behavior, on=["subject", "date"], how="left")
    else:
        df_merge = df_mean_neural_score_stim.copy()
    # sort by subject and date
    df_merge = df_merge.sort_values(by=["subject", "date"])

    l_train = []
    l_test = []

    for sub in df_merge["subject"].unique():
        df_sub_train = df_merge[df_merge["subject"] == sub]
        dates_sub = df_sub_train["date"].unique()
        for date in dates_sub:
            df_sub_date = df_sub_train[df_sub_train["date"] == date]
            mean_prev_ybocs = df_sub_train[df_sub_train["date"] < date]["YBOCS II Total Score"].mean()
            # the next date is always the test date, if it exists
            next_date = df_sub_train[df_sub_train["date"] > date]["date"].min()
            if pd.isna(next_date):
                continue
            df_sub_date["y_test_date"] = next_date
            df_sub_date["YBOCS II Total Score_test"] = df_sub_train[df_sub_train["date"] == next_date]["YBOCS II Total Score"].values[0]
            df_sub_date["YBOCS II Total Score_mean_prev"] = mean_prev_ybocs
            l_train.append(df_sub_date)

    train_df = pd.concat(l_train, ignore_index=True)

    # sort by subject, date
    train_df = train_df.sort_values(by=["subject", "date"])

    # iterate now through the dataframe, but split the training by time, training for a test subject can include training data but only up to that date
    l_out = []
    for sub in tqdm(train_df["subject"].unique()):
        df_sub_train = train_df[train_df["subject"] == sub]
        df_othersub_train = train_df[train_df["subject"] != sub]
        dates_sub = df_sub_train["date"].unique()
        for date in dates_sub:
            df_sub_date = df_sub_train[df_sub_train["date"] == date]
            prev_dates = df_sub_train[df_sub_train["date"] < date]
            if prev_dates.empty:
                continue
            X_train = pd.concat([prev_dates, df_othersub_train], ignore_index=True)
            y_train = X_train["YBOCS II Total Score_test"]
            X_train = X_train.drop(columns=["YBOCS II Total Score_test"])
            X_test = df_sub_date.drop(columns=["YBOCS II Total Score_test"])
            y_test = df_sub_date["YBOCS II Total Score_test"].iloc[0]
            # drop labels with NaN in y_train
            X_train = X_train[~y_train.isna()]
            y_train = y_train[~y_train.isna()]
            if y_test is None or pd.isna(y_test):
                continue
            model = CatBoostRegressor(cat_features=["subject"], verbose=0)
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            # get sample of that subject with the most recent date
            X_train_most_recent = X_train[X_train["subject"] == sub].sort_values(by="date", ascending=False).iloc[0]
            mean_predictor = X_train_most_recent["YBOCS II Total Score_mean_prev"]
            prev_ybocs_pred = X_train_most_recent["YBOCS II Total Score"]
            l_out.append({"subject": sub, "date": date, "true": y_test, "pred": pred[0], "mean_predictor": mean_predictor, "prev_ybocs_pred": prev_ybocs_pred})

    res_df = pd.DataFrame(l_out)
    res_df["INCLUDE_STIM"] = INLCUDE_STIM
    res_df["INCLUDE_BEHAV"] = INCLUDE_BEHAV
    res_df.to_csv(f"merge_stim_behav_neural/out/results_STIM_{INLCUDE_STIM}_BEHAV_{INCLUDE_BEHAV}.csv", index=False)

    res_df["error"] = res_df["true"] - res_df["pred"]
    res_df["mse"] = res_df["error"]**2
    res_df["mae"] = res_df["error"].abs()
    mse_per_subject = res_df.groupby("subject")["mse"].mean()
    mae_per_subject = res_df.groupby("subject")["mae"].mean()

    # compute mse for mean predictor and previous ybocs predictor
    res_df["error_mean_predictor"] = res_df["true"] - res_df["mean_predictor"]
    res_df["mse_mean_predictor"] = res_df["error_mean_predictor"]**2
    res_df["error_prev_ybocs_pred"] = res_df["true"] - res_df["prev_ybocs_pred"]
    res_df["mse_prev_ybocs_pred"] = res_df["error_prev_ybocs_pred"]**2
    mse_mean_predictor_per_subject = res_df.groupby("subject")["mse_mean_predictor"].mean()
    mse_prev_ybocs_pred_per_subject = res_df.groupby("subject")["mse_prev_ybocs_pred"].mean()
    mae_mean_predictor_per_subject = res_df.groupby("subject")["error_mean_predictor"].apply(lambda x: x.abs().mean())
    mae_prev_ybocs_pred_per_subject = res_df.groupby("subject")["error_prev_ybocs_pred"].apply(lambda x: x.abs().mean())

    corr_per_subject = res_df.groupby("subject").apply(lambda x: x["true"].corr(x["pred"]))
    corr_mean_predictor_per_subject = res_df.groupby("subject").apply(lambda x: x["true"].corr(x["mean_predictor"]))
    corr_prev_ybocs_pred_per_subject = res_df.groupby("subject").apply(lambda x: x["true"].corr(x["prev_ybocs_pred"]))

    df_results = pd.DataFrame({
        "mse": mse_per_subject,
        "mae": mae_per_subject,
        "mse_mean_pred": mse_mean_predictor_per_subject,
        "mae_mean_pred": mae_mean_predictor_per_subject,
        "mse_prev_ybocs": mse_prev_ybocs_pred_per_subject,
        "mae_prev_ybocs": mae_prev_ybocs_pred_per_subject,
        "corr": corr_per_subject,
        "corr_mean_pred": corr_mean_predictor_per_subject,
        "corr_prev_ybocs": corr_prev_ybocs_pred_per_subject,
        "subject": mse_per_subject.index
    })
    df_results.to_csv(f"merge_stim_behav_neural/out/results_summary_STIM_{INLCUDE_STIM}_BEHAV_{INCLUDE_BEHAV}.csv", index=False)

    plt.figure(figsize=(10, 7))
    for col_idx, col in enumerate(["mae", "mse", "corr"]):
        plt.subplot(1, 3, col_idx+1)
        cols_ = [c for c in df_results.columns if col in c]
        df_plt = df_results[["subject"] + cols_].melt(id_vars=["subject"], value_vars=cols_, var_name="metric", value_name="value")
        sns.boxplot(data=df_plt, x="metric", y="value", showmeans=True)
        sns.swarmplot(data=df_plt, x="metric", y="value", color=".25")
        # put mean and std for each group in the title
        means = df_plt.groupby("metric")["value"].mean()
        stds = df_plt.groupby("metric")["value"].std()
        plt.title(f"{col}:" + " ".join([f"{m}: {means[m]:.2f} ± {stds[m]:.2f}\n" for m in means.index]))
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"merge_stim_behav_neural/figures/boxplot_STIM_{INLCUDE_STIM}_BEHAV_{INCLUDE_BEHAV}.pdf")

    plt.figure(figsize=(10, 7))
    for sub_idx, sub in enumerate(res_df["subject"].unique()):
        plt.subplot(3, 3, sub_idx+1)
        df_sub = res_df[res_df["subject"] == sub]
        plt.plot(df_sub["date"], df_sub["true"], label=f"{sub} true", marker="o")
        plt.plot(df_sub["date"], df_sub["pred"], label=f"{sub} pred", marker="x")
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.ylabel("YBOCS II Total Score")
        plt.title(f"{sub}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"merge_stim_behav_neural/figures/time_pred_STIM_{INLCUDE_STIM}_BEHAV_{INCLUDE_BEHAV}.pdf")

    print()