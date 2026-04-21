#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script: pronoun_embedding_trajectory_numerical.py

Purpose:
Visualize and print how a pronoun's embedding moves toward its correct referent across transformer layers.

Outputs:
1. Numerical L2 distances for each layer
2. Graphical plot of distances across layers
3. Optional PCA plot for 2D trajectory visualization
"""

import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "bert-base-uncased"
SENTENCE = "The man spoke to the woman because he was tired."
TARGET_PRONOUN = "he"
CANDIDATES = ["man", "woman"]

# ---------------------------
# INITIALIZATION
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
model.eval()

# ---------------------------
# TOKENIZATION
# ---------------------------
tokens = tokenizer.tokenize(SENTENCE)
token_ids = tokenizer.encode(SENTENCE, return_tensors="pt")

def find_token_index(target):
    for idx, t in enumerate(tokens):
        if target.lower() in t.lower():
            return idx
    return None

pronoun_idx = find_token_index(TARGET_PRONOUN)
candidate_indices = [find_token_index(word) for word in CANDIDATES]

if pronoun_idx is None or any(idx is None for idx in candidate_indices):
    raise ValueError("Could not find all target tokens in the tokenized sentence.")

# ---------------------------
# MODEL FORWARD PASS
# ---------------------------
with torch.no_grad():
    outputs = model(token_ids)
    hidden_states = torch.stack(outputs.hidden_states).squeeze(1)  # (num_layers, seq_len, hidden_dim)

# ---------------------------
# EXTRACT EMBEDDINGS
# ---------------------------
pronoun_embs = hidden_states[:, pronoun_idx, :]
candidate_embs = [hidden_states[:, idx, :] for idx in candidate_indices]

# ---------------------------
# CALCULATE L2 DISTANCES
# ---------------------------
layer_nums = list(range(hidden_states.shape[0]))
distances = []
for cand_emb in candidate_embs:
    dist = torch.norm(pronoun_embs - cand_emb, dim=1)  # layer-wise L2 distance
    distances.append(dist.numpy())

# ---------------------------
# PRINT NUMERICAL OUTPUT
# ---------------------------
print(f"Layer-wise L2 distances for pronoun '{TARGET_PRONOUN}':")
for i, layer in enumerate(layer_nums):
    layer_str = f"Layer {i}: "
    for j, word in enumerate(CANDIDATES):
        layer_str += f"{word}={distances[j][i]:.4f}  "
    print(layer_str)

# ---------------------------
# PLOT DISTANCES ACROSS LAYERS
# ---------------------------
plt.figure(figsize=(8,5))
for i, word in enumerate(CANDIDATES):
    plt.plot(layer_nums, distances[i], label=f"Distance to '{word}'", marker='o')
plt.xlabel("Layer")
plt.ylabel("L2 Distance")
plt.title(f"Pronoun '{TARGET_PRONOUN}' Embedding Trajectory")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# OPTIONAL: PCA VISUALIZATION
# ---------------------------
all_embs = torch.stack([pronoun_embs] + candidate_embs)
num_layers = hidden_states.shape[0]
all_embs_2d = all_embs.view(-1, all_embs.shape[-1])

pca = PCA(n_components=2)
all_embs_2d = pca.fit_transform(all_embs_2d.numpy())

plt.figure(figsize=(8,5))
for i, word in enumerate([TARGET_PRONOUN] + CANDIDATES):
    x = all_embs_2d[i*num_layers:(i+1)*num_layers, 0]
    y = all_embs_2d[i*num_layers:(i+1)*num_layers, 1]
    plt.plot(x, y, marker='o', label=word)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Pronoun and Candidate Embeddings Across Layers (PCA)")
plt.legend()
plt.grid(True)
plt.show()
