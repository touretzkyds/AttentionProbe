import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# === Load model ===
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    output_attentions=True,
    output_hidden_states=True
)
model.eval()

# === Input sentence ===
sentence = "The woman showed the man her jacket."
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("\nTOKENS:", tokens)

# === Run encoder ===
with torch.no_grad():
    outputs = model.encoder(
        input_ids=inputs["input_ids"],
        output_attentions=True,
        output_hidden_states=True
    )

hidden_states = outputs.hidden_states  # (13 layers = embeddings + 12 blocks)
attentions = outputs.attentions        # (12 layers)

# === Find token indices ===
def find_index(tok_list, word):
    for i, t in enumerate(tok_list):
        if t.replace("▁", "") == word:
            return i
    return None

idx_man = find_index(tokens, "man")
idx_woman = find_index(tokens, "woman")
idx_pron = find_index(tokens, "her")

print(f"Token indices → man={idx_man}, woman={idx_woman}, pronoun={idx_pron}")

# === Cosine similarity (pronoun ↔ nouns) ===
cos = torch.nn.functional.cosine_similarity
sim_data = []

for l, h in enumerate(hidden_states):
    if l == 0:  # layer 0 = token embeddings
        continue
    pron_vec = h[0, idx_pron, :]
    man_vec = h[0, idx_man, :]
    woman_vec = h[0, idx_woman, :]

    sim_man = cos(pron_vec, man_vec, dim=0).item()
    sim_woman = cos(pron_vec, woman_vec, dim=0).item()
    sim_data.append((l, sim_man, sim_woman))

sim_df = pd.DataFrame(sim_data, columns=["Layer", "Pronoun→Man", "Pronoun→Woman"])
print("\n=== Cosine Similarity (Pronoun vs Nouns) ===")
print(sim_df.to_string(index=False))

# === Attention (pronoun → nouns) ===
attn_data = []
for l, attn in enumerate(attentions):
    avg_attn = attn.mean(1)[0]  # mean over heads → (seq_len, seq_len)
    attn_man = avg_attn[idx_pron, idx_man].item()
    attn_woman = avg_attn[idx_pron, idx_woman].item()
    attn_data.append((l, attn_man, attn_woman))

attn_df = pd.DataFrame(attn_data, columns=["Layer", "Pronoun→Man", "Pronoun→Woman"])
print("\n=== Attention Weights (Pronoun to Nouns) ===")
print(attn_df.to_string(index=False))

# === Plot cosine similarities ===
plt.figure(figsize=(8,5))
plt.plot(sim_df["Layer"], sim_df["Pronoun→Man"], label="Pronoun–Man", marker="o")
plt.plot(sim_df["Layer"], sim_df["Pronoun→Woman"], label="Pronoun–Woman", marker="s")
plt.xlabel("Layer")
plt.ylabel("Cosine Similarity")
plt.title(f"Contextual Similarity Across Layers\n{sentence}")
plt.legend()
plt.grid(True)
plt.show()

# === Plot attention heatmap ===
attn_matrix = attn_df[["Pronoun→Man", "Pronoun→Woman"]].to_numpy()
plt.figure(figsize=(6,5))
sns.heatmap(attn_matrix, annot=True, fmt=".3f",
            xticklabels=["man", "woman"],
            yticklabels=[f"L{l}" for l in attn_df["Layer"]],
            cmap="magma")
plt.xlabel("Noun")
plt.ylabel("Layer")
plt.title(f"Attention from Pronoun '{tokens[idx_pron]}' to Nouns")
plt.show()
