import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import pdf
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch
from tqdm import tqdm
import os
from scipy import stats

df = pd.read_csv("else/df_features_rick_3days_normed.csv")

with open("else/time_periods_3days.pkl", "rb") as f:
    time_periods = pickle.load(f)
subs = df["subject"].unique()

pdf_plt = PdfPages("else/plot_spectra_rick_3days.pdf")
for sub_idx, sub in enumerate(subs):
    df_sub = df.query("subject == @sub")
    plt.figure()
    plt.suptitle(sub)
    for lr_idx, hem in enumerate(["Pxx_left", "Pxx_right"]):
        plt.subplot(1, 2, lr_idx + 1)
        if hem == "Pxx_left":
            plt.title(("L"))
        else:
            plt.title(("R"))
        resp_Pxx = []
        nonresp_Pxx = []
        preDBS_Pxx = []
        for idx, row in df_sub.iterrows():
            Pxx = row[hem]
            if Pxx is np.nan:
                continue
            Pxx = np.fromstring(Pxx.strip("[]"), sep=" ").tolist()
            #Pxx = np.log(Pxx)
            responder_ = row["response"]
            if responder_ == "Responder":
                color_ = "blue"
                resp_Pxx.append(Pxx)
            elif responder_ == "Non-Responder":
                color_ = "red"
                nonresp_Pxx.append(Pxx)
            elif responder_ == "Pre-DBS":
                color_ = "black"
                preDBS_Pxx.append(Pxx)
            plt.plot(time_periods, Pxx, alpha=0.1, color=color_)
        if nonresp_Pxx:
            nonresp_Pxx = np.array(nonresp_Pxx)
            mean_nonresp_Pxx = np.nanmean(nonresp_Pxx, axis=0)
            plt.plot(time_periods, mean_nonresp_Pxx, label="Non-Responder", color="lightcoral", linewidth=2)

        if resp_Pxx:
            resp_Pxx = np.array(resp_Pxx)
            mean_resp_Pxx = np.nanmean(resp_Pxx, axis=0)
            plt.plot(time_periods, mean_resp_Pxx, label="Responder", color="deepskyblue", linewidth=2)
        if preDBS_Pxx:
            preDBS_Pxx = np.array(preDBS_Pxx)
            mean_preDBS_Pxx = np.nanmean(preDBS_Pxx, axis=0)
            plt.plot(time_periods, mean_preDBS_Pxx, label="Pre-DBS", color="gray", linewidth=2)
        plt.xscale("log")
        #plt.xlim(1, 40)
        plt.xlabel("Time period [h]")
        plt.ylabel("Power [a.u.]")
        # despine right and top
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.legend()
    plt.tight_layout()
    pdf_plt.savefig(plt.gcf())
    plt.close()
pdf_plt.close()
