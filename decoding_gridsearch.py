import pandas as pd
from sklearn import metrics, linear_model
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from scipy import stats
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import numpy as np
import seaborn as sns
from decoder import compute_ml
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib



df_features = pd.read_csv("features_prep_combined_wide.csv")

CLASSIFICATION = False

corrs = []
p_val = []
ba_ = []
true_preds = []
preds = []

col_score = "YBOCS II Total Score"  # or "YBOCS II-Compulsions Sub-score" or "YBOCS II Total Score"
ML_models = ["NeuralNet", "SVR_rbf", "RF", "Linear", "XGB", "SVR_linear"]
#feature_mods = ["all", "fooof", "psd", "Sharpwave", "burst", "raw", "Hjorth", "burst_low_f"]
feature_mods = ["fft_only", "fft_psd", "Hjorth", "Sharpwave", "fooof", "coherence", "burst", "all", "alpha", "beta", "delta", "gamma", "theta", "burst_amplitude", "burst_duration"]
locs = [ "SC_L_", "C_L_1_", "C_L_2_", "C_R_1_", "C_R_2_", "SC_R_"]
# compute_ml(df_features, col_score=col_score,
#            feature="burst",
#            model_type="XGB",
#            loc="SC_L")

#compute_ml(df_features, col_score, "Hjorth", "XGB", "SC_L_")
#compute_ml(df_features, col_score, "fft_only", "XGB","SC_L_")
#compute_ml(df_features, col_score, "fft_psd", "XGB","SC_L_" )
#compute_ml(df_features, col_score, "all", "Linear", "C_R_1_")
#compute_ml(df_features, col_score, "all", "Linear", "SC_L_")

# results = []
# for model_type in tqdm(ML_models):
#     for loc in locs:
#         for feature_mod in feature_mods[1:]:
#             per_ = compute_ml(df_features, col_score, feature_mod, model_type, loc)
#             if per_ is not None:
#                 results.append(per_)

tasks = [
    delayed(compute_ml)(df_features, col_score, feature_mod, model_type, loc)
    for model_type in ML_models
    for loc in locs
    for feature_mod in feature_mods
]

with tqdm_joblib(tqdm(desc="Running ML models", total=len(tasks))):
    results = Parallel(n_jobs=50)(tasks)

df_results = pd.concat([pd.DataFrame(res) for res in results], ignore_index=True)

#df_results = pd.DataFrame(list(np.ravel(results)))
df_results["model_type"] = df_results["model"] + "_" + df_results["feature"]

df_results.to_csv("results_ml.csv", index=False)
