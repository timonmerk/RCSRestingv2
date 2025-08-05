import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

PLOT_ = False

df_scores = pd.read_csv("map_scores/scores_date_mapped.csv")
df_scores["date"] = pd.to_datetime(df_scores["date"])
#replace for sub aDBS009 in column IUS Total Score the value 8/ with 8
df_scores["IUS Total Score"] = df_scores["IUS Total Score"].replace("8/", "8")
all_score_names= df_scores.columns.tolist()[:-3]
for col in all_score_names:
    df_scores[col] = df_scores[col].replace("Pending", np.nan)
    df_scores[col] = df_scores[col].replace("95 Incomplete", np.nan)

corr_matrices = []
for i, sub in enumerate(np.sort(df_scores["subject"].unique())):
    df_q = df_scores.query("subject == @sub").reset_index(drop=True).sort_values(by="date")        
    
    #df_q = df_q.dropna(subset=all_score_names)
    corr_matrix = df_q[all_score_names].corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
    corr_matrix_add = df_q[all_score_names].corr()
    corr_matrices.append(corr_matrix_add)

    if PLOT_:
        plt.figure(figsize=(15, 15))
        sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
        plt.title(f"Correlation matrix for {sub}")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"figures/scores/corr_matrix_{sub}.pdf")

if PLOT_:
    plt.figure(figsize=(15, 15))
    plt.imshow(np.nanmean(corr_matrices, axis=0), cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(ticks=range(len(all_score_names)), labels=all_score_names, rotation=90)
    plt.yticks(ticks=range(len(all_score_names)), labels=all_score_names)
    plt.title("Average Correlation Matrix across Subjects")
    plt.tight_layout()
    plt.savefig("figures/scores/corr_matrix_avg.pdf")

# PCA Analysis
# Step 1: Eigen-decomposition
corr_matrix = np.nanmean(np.array(corr_matrices), axis=0)
eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

plt.figure(figsize=(6, 4))
plt.plot(explained_variance_ratio, marker='o', linestyle='-')
plt.title("Explained Variance Ratio by Eigenvalues")
plt.xlabel("Principal Component Index")
plt.ylabel("Explained Variance Ratio")
plt.savefig("figures/pca_explained_variance_ratio.pdf")
plt.show(block=True)

# make a bar plot of the first 3 eigenvectores
plt.figure(figsize=(12, 7))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.barh(range(len(eigenvectors[:, i])), eigenvectors[:, i])
    plt.title(f"#{i+1}")
    if i == 0:
        plt.yticks(ticks=range(len(all_score_names)), labels=all_score_names)
    if i != 0:
        plt.yticks([])
        # turn off all labels
        plt.tick_params(axis='y', which='both', left=False, labelleft=False)
    # turn off axis spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if i != 0:
        plt.gca().spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/pca_eigenvectors_first3.pdf")


for column_plt in df_scores.columns[:-3]:
    plt.figure(figsize=(10, 20))
    plt_idx = 1
    for i, sub in enumerate(np.sort(df_scores["subject"].unique())):
        plt.subplot(len(df_scores["subject"].unique()), 1, plt_idx)
        plt_idx += 1
        df_q = df_scores.query("subject == @sub").reset_index(drop=True).sort_values(by="date")
        df_q = df_q.dropna(subset=[column_plt])
        plt.plot(df_q["date"], df_q[column_plt], marker='o', label=sub)
        plt.xticks(rotation=45)
        plt.title(sub)
        plt.ylabel(column_plt)
        plt.xlabel("Date")
    plt.tight_layout()
    plt.suptitle(column_plt)
    plt.savefig(f"figures/scores/{column_plt}.pdf")
    #plt.show(block=True)

    # make instead one plot with all subjects
    plt.figure(figsize=(10, 5))
    for i, sub in enumerate(np.sort(df_scores["subject"].unique())):
        df_q = df_scores.query("subject == @sub").reset_index(drop=True).sort_values(by="date")
        df_q = df_q.dropna(subset=[column_plt])
        plt.plot(df_q["date"], df_q[column_plt], marker='o', label=sub)
    plt.xticks(rotation=45)
    plt.title("Scores over time")
    plt.ylabel(column_plt)
    plt.xlabel("Date")
    plt.legend()
    plt.suptitle(column_plt)
    plt.savefig(f"figures/scores/all_sub_{column_plt}.pdf")
    #plt.tight_layout()