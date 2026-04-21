import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- LOAD YOUR COSINE RESULTS ----
df = pd.read_csv("/home/navya/AttentionProbe/cosine_results_no_correcteness.csv")

# Select only cosine columns
cos_cols = ["cos_his_man", "cos_his_woman", "cos_her_man", "cos_her_woman"]
cos_df = df[cos_cols]

# Convert to matrix for heatmap
cos_matrix = cos_df.to_numpy()

plt.figure(figsize=(10, 6))

sns.heatmap(
    cos_matrix,
    annot=True,
    cmap="viridis",
    xticklabels=cos_cols,
    yticklabels=df["row_index"],
    cbar=True
)

plt.title("Cosine Similarity Heatmap: Pronoun ↔ Subject")
plt.xlabel("Pair")
plt.ylabel("Sentence Row Index")
plt.tight_layout()
plt.show()
