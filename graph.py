#load imports
print("starting")
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
print("imports done")

#loading the model
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
print("Tokenizer loaded")
model = T5ForConditionalGeneration.from_pretrained(model_name)
print("Model loaded")

#getting the embedding library
embedding_matrix = model.shared.weight.detach()
print("Matrix detached", embedding_matrix.shape)

#computing cosine similarity
def get_similarity(word1, word2):
    #tokenize words given
    tok1 = tokenizer.tokenize(word1)
    tok2 = tokenizer.tokenize(word2)

    if len(tok1) > 1:
        print(f"Warning: {word1} tokenizes as {tok1}")
    if len(tok2) > 1:
        print(f"Warning: {word2} tokenizes as {tok2}")

    #converting tokens to the ids found in the dictionary
    id1 = tokenizer.convert_tokens_to_ids(tok1)[0]
    id2 = tokenizer.convert_tokens_to_ids(tok2)[0]

    #get the embedding vectors
    vec1 = embedding_matrix[id1]
    vec2 = embedding_matrix[id2]

    # cosine similarity
    cos = cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    return cos, tok1[0], tok2[0]

#collects input from the user
def collect_pairs(category_name):
    print(f"\n=== Enter 5 {category_name} word pairs ===")
    sims = []
    for i in range(5):
        w1 = input(f"Pair {i+1} - word 1: ").strip()
        w2 = input(f"Pair {i+1} - word 2: ").strip()
        cos, t1, t2 = get_similarity(w1, w2)
        print(f"  tokenized: {t1} vs {t2} → cosine similarity = {cos:.4f}\n")
        sims.append(cos)
    return sims

def process_pairs(word_pairs):
    sims = []
    for (w1, w2) in word_pairs:
        cos, t1, t2 = get_similarity(w1, w2)
        print(f"  tokenized: {t1} vs {t2} → cosine similarity = {cos:.4f}\n")
        sims.append(cos)
    return sims

# Pre-set word pairs
semantic_word_pairs = (("pants", "trousers"), ("strange", "weird"), ("begin", "start"), ("buy", "purchase"), ("lunch", "dinner"))
orthographic_word_pairs = (("read", "red"), ("whale", "whole"), ("desk", "risk"), ("trap", "tram"), ("seat", "seal"))
phonetic_word_pairs = (("carpet", "knife"), ("ball", "tower"), ("telephone", "witch"), ("table", "song"), ("height", "car"))

# Compute similarities
semantic_sims = process_pairs(semantic_word_pairs)
orthographic_sims = process_pairs(orthographic_word_pairs)
phonetic_sims = process_pairs(phonetic_word_pairs)

print("similarites collected")

import numpy as np

plt.figure(figsize=(12,7))

# Plot the points
plt.scatter([1]*len(semantic_sims), semantic_sims, color='blue', s=80, label="Semantic")
plt.scatter([2]*len(orthographic_sims), orthographic_sims, color='green', s=80, label="Orthographic")
plt.scatter([3]*len(phonetic_sims), phonetic_sims, color='red', s=80, label="Phonetic")

# ---- FUNCTION TO ADD READABLE LABELS ----
def label_points(x_pos, sims, pairs, color):
    # jitter offsets to separate labels vertically
    offsets = np.linspace(-0.03, 0.03, len(sims))
    
    for i, ((w1, w2), y, dy) in enumerate(zip(pairs, sims, offsets)):
        label = f"{w1}-{w2}"
        plt.text(
            x_pos + 0.05,     # horizontal offset to the right
            y + dy,           # vertical offset so they don’t overlap
            label,
            fontsize=9,
            color=color
        )

# Add labels for each category
label_points(1, semantic_sims, semantic_word_pairs, "blue")
label_points(2, orthographic_sims, orthographic_word_pairs, "green")
label_points(3, phonetic_sims, phonetic_word_pairs, "red")

plt.xticks([1,2,3], ['Semantic', 'Orthographic', 'Phonetic'])
plt.ylabel("Cosine Similarity")
plt.title("Embedding Similarity Across User-Provided Word Categories")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

