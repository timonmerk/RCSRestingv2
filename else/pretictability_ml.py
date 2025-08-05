from turtle import pos
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.utils.multiclass import unique_labels

from sklearn import metrics, linear_model
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm
import numpy as np


def run_ml_modality(df, mod, model_arch, return_predictions=False):
    subs = df["subject"].unique()
    df_res = []
    y_pred_subs = []
    y_true_subs = []
    for sub in subs:
        X_train = df.query("subject != @sub").drop(columns=["subject", "response", "date"])
        X_test = df.query("subject == @sub").drop(columns=["subject", "response", "date"])
        y_test = df.query("subject == @sub")["response"]
        cols_use = [c for c in X_train.columns if mod in c]
        if mod == "all":
            cols_use = X_train.columns
        if mod == "Hjorth_left":
            cols_use = [c for c in X_train.columns if "Hjorth" in c and "left" in c]
        elif mod == "Hjorth_right":
            cols_use = [c for c in X_train.columns if "Hjorth" in c and "right" in c]
        elif mod == "power_left":
            cols_use = [c for c in X_train.columns if "power" in c and "left" in c]
        elif mod == "power_right":
            cols_use = [c for c in X_train.columns if "power" in c and "right" in c]
        elif mod == "24h_left_normed_and_delta_lfp_dayR2":
            cols_use = ["delta_lfp_left_OvER_interpolate_dayR2", "power_24h_left_normed"]

        X_train = X_train[cols_use]
        X_test = X_test[cols_use]
        if X_train.empty or X_test.empty:
            print(f"Skipping {sub} due to empty train or test set")
            continue

        # make Pre-DBS 0 and Responder 1
        y_train = df.query("subject != @sub")["response"].replace({"Pre-DBS": 0, "Responder": 1, "Non-Responder": 0})
        y_test = y_test.replace({"Pre-DBS": 0, "Responder": 1, "Non-Responder": 0})
        #if y_test.unique().size == 1:
        #    print(f"{sub} {y_test.unique()}")
        #    continue

        print(f"{sub} {y_test.value_counts()}")

        if model_arch == "xgb":
            neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
            scale_pos_weight = neg / pos
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
        elif model_arch == "logistic":
            model = linear_model.LogisticRegression(class_weight='balanced')
        # remove rows that have only NaN values
        idx_nan_all = X_train.index[X_train.isna().any(axis=1)]
        X_train = X_train.drop(index=idx_nan_all)
        y_train = y_train.drop(index=idx_nan_all)

        idx_nan_test = X_test.index[X_test.isna().any(axis=1)]
        X_test = X_test.drop(index=idx_nan_test)
        y_test = y_test.drop(index=idx_nan_test)
        if y_test.shape[0] == 0:
            print(f"Skipping {sub} due to empty test set")
            continue
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error fitting model for {sub}: {e}")
            continue
        y_pred = model.predict(X_test)
        # get the AUC
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        labels_present = np.unique(np.concatenate([y_test, y_pred]))

        # Force the confusion matrix to be 2x2
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tpr, tnr, auc, ba = None, None, None, None

        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        tnr = tn / (tn + fp) if (tn + fp) > 0 else None
        if len(labels_present) == 1 and 1 in labels_present:
            tnr = None
        elif len(labels_present) == 1 and 0 in labels_present:
            tpr = None

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_proba)
            ba = metrics.balanced_accuracy_score(y_test, y_pred)

        df_res.append({"subject": sub, "auc": auc, "tnr" : tnr, "tpr" : tpr, "ba": ba})
        y_pred_subs.append(y_pred_proba)
        y_true_subs.append(y_test.values)
    df_res = pd.DataFrame(df_res)
    df_res["mod"] = mod
    df_res["arch"] = model_arch
    if return_predictions:
        return df_res, y_pred_subs, y_true_subs, subs
    return df_res

modalities = ["delta_lfp_left_OvER_interpolate_dayR2", "24h_left_normed_and_delta_lfp_dayR2", 
              "24h_left_normed", "left", "right", "power", "Hjorth", "Hjorth_activity_left", 
              "Hjorth_activity_right", "Hjorth_mobility_left", "Hjorth_mobility_right", 
              "Hjorth_complexity_left", "Hjorth_complexity_right", "all",
              "power_left", "power_right", "Hjorth_left", "Hjorth_right",
               "12h_left", "24h_left", "12h_right", "24h_right", "normed", "6h_left", "6h_right", "12h_left_normed",
               "12h_right_normed", "24h_right_normed", "6h_left_normed", "6h_right_normed", "power_left_sum", "power_right_sum",]

archs = ["logistic", "xgb"] # 

df_comp_out = []
for time_range in np.arange(1, 16, 1):
    df = pd.read_csv(f"else/df_features_rick_{time_range}days_normed_by_sum.csv")
    df.drop(columns=["Unnamed: 0", "Pxx_left", "Pxx_right"], inplace=True)
    subs_remove = ["B009", "B0014", "B020"]
    df = df.query("subject not in @subs_remove")

    feature_cols = [c for c in df.columns if c not in ["subject", "response", "date", "delta_lfp_left_OvER_interpolate_dayR2"]]
    NORM_BY_PREDBS = True
    if NORM_BY_PREDBS:
        for sub in df["subject"].unique():
            df_sub = df.query("subject == @sub")
            pre_dbs_mean = df_sub.query("response == 'Pre-DBS'")[feature_cols].mean()
            df.loc[df["subject"] == sub, feature_cols] -= pre_dbs_mean

    # df_res, y_pred_subs, y_true_subs, subs = run_ml_modality(df, 'delta_lfp_left_OvER_interpolate_dayR2',
    #                                                          "logistic", return_predictions=True)
    
    # remove the PreDBS rows
    df = df.query("response != 'Pre-DBS'")
    for mod in tqdm(modalities):
        for arch in archs:
            df_res = run_ml_modality(df, mod, arch)
            df_res["num_days"] = time_range
            df_comp_out.append(df_res)

df_comp = pd.concat(df_comp_out, ignore_index=True)
df_comp["arch_mod_days"] = df_comp["arch"] + "_" + df_comp["mod"] + "_" + df_comp["num_days"].astype(str)
df_comp.to_csv("else/df_comp_ml_LRXGB.csv", index=False)

# df_best = df_comp.groupby("arch_mod_days")[["tnr", "tpr"]].mean()
# df_best["mean_tnr_tpr"] = (df_best["tnr"] + df_best["tpr"]) / 2

# # get max mean_tnr_tpr
# best_arch_mod_days = df_best["mean_tnr_tpr"].idxmax()

# df_comp.query("arch_mod_days == 'logistic_24h_left_normed_and_delta_lfp_dayR2_4'")  #  @best_arch_mod_days
# df_comp.query("arch_mod_days == @best_arch_mod_days")


# df_comp.groupby("arch_mod_days")["mean_tnr_tpr"].mean().sort_values(ascending=False)

# order_by_mean_arch_mod = df_comp.groupby("arch_mod_days")["auc"].mean().sort_values(ascending=False).index.tolist()
# best_auc = df_comp.groupby("arch_mod_days")["auc"].mean().max()
# best_ba = df_comp.groupby("arch_mod_days")["ba"].mean().max()




# df_comp.query("arch_mod_days == @order_by_mean_arch_mod[0]")

# plt.figure(figsize=(10, 5))
# for plt_idx, plt_label in enumerate(["auc", "ba", "tpr", "tnr"]):
#     plt.subplot(1, 4, plt_idx + 1)
#     sns.boxplot(data=df_comp, x="arch_mod_days", y=plt_label, order=order_by_mean_arch_mod[:10], showmeans=True, showfliers=False)
#     sns.swarmplot(data=df_comp, x="arch_mod_days", y=plt_label, color=".25", alpha=0.5, order=order_by_mean_arch_mod[:10])
#     plt.xticks(rotation=90)
#     plt.title(plt_label)
# plt.suptitle(f"Best AUC: {best_auc:.3f}, Best BA: {best_ba:.3f}")
# plt.tight_layout()



# B009, B0014, B020