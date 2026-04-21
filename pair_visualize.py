"""
Visual comparison for a pair of pronoun-resolution sentences.

What it does:
- Loads CSV and T5 encoder
- Picks two sentences by cleaned-row index (IDX_A, IDX_B)
- Computes:
    - cos(his, man) and cos(his, woman) in each sentence
    - Cross-sentence cosines for man, woman, his, jacket
    - Attention from pronoun -> man/woman (per head) in last layer
- VISUALIZES:
    - Bar chart: pronoun→subject cosine similarities (A vs B)
    - Bar charts: pronoun→man and pronoun→woman attention per head
      for heads whose attention changed the most between A and B
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5EncoderModel

# =========================
# CONFIG
# =========================

CSV_PATH = "/home/navya/Attention Probe Results - Sheet1.csv"
MODEL_NAME = "google/flan-t5-large"

# Choose the two sentence row indices in the CLEANED df
IDX_A = 18   # e.g. "The woman showed the man his jacket..."
IDX_B = 28   # e.g. "The man was shown his jacket by a woman..."

TARGET_WORDS = ["man", "woman", "his", "her", "jacket"]
CONTROL_WORD = "jacket"

ERROR_STATES = {
    "Incorrect Answer",
    "Inconsistent with one or more runs",
    "Ambigious sentence",
    "Hallucination ",
    "Abnormality",
}

# =========================
# MODEL & TOKENIZER
# =========================

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5EncoderModel.from_pretrained(
    MODEL_NAME,
    output_attentions=True,
    output_hidden_states=True,
)
model.eval()
print("Model loaded.")


# =========================
# HELPER FUNCTIONS
# =========================

def get_tokens(sentence):
    enc = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return enc, tokens

def find_word_indices(tokens, word):
    word = word.lower()
    idxs = []
    for i, tok in enumerate(tokens):
        clean = tok.replace("▁", "").lower()
        if clean == word:
            idxs.append(i)
    return idxs

def get_last_hidden_and_attentions(sentence):
    enc, tokens = get_tokens(sentence)
    with torch.no_grad():
        out = model(**enc)
    last_hidden = out.last_hidden_state[0]   # [seq, hidden_dim]
    attentions = out.attentions             # list[layers] [1, heads, seq, seq]
    return tokens, last_hidden, attentions

def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

def get_word_embeddings(sentence):
    tokens, last_hidden, attentions = get_last_hidden_and_attentions(sentence)
    word_embs = {}
    positions = {}

    for w in TARGET_WORDS:
        idxs = find_word_indices(tokens, w)
        if idxs:
            positions[w] = idxs
            word_embs[w] = last_hidden[idxs].mean(dim=0)
    return tokens, word_embs, positions, attentions

def get_pronoun_attention_to_subjects(tokens, attentions, positions):
    """
    Returns a dict per head:
      head -> (frac_to_man, frac_to_woman)
    using the last encoder layer.
    """
    last_layer = attentions[-1][0]  # [heads, seq, seq]

    his_idxs = positions.get("his", [])
    her_idxs = positions.get("her", [])
    pronoun_idxs = his_idxs + her_idxs

    man_idxs = positions.get("man", [])
    woman_idxs = positions.get("woman", [])

    results = {}
    if not pronoun_idxs:
        return results

    for head in range(last_layer.shape[0]):
        att_to_man = 0.0
        att_to_woman = 0.0
        total = 0.0

        for p in pronoun_idxs:
            vec = last_layer[head, p]  # [seq]
            total += float(vec.sum())
            if man_idxs:
                att_to_man += float(vec[man_idxs].sum())
            if woman_idxs:
                att_to_woman += float(vec[woman_idxs].sum())

        if total > 0:
            frac_man = att_to_man / total
            frac_woman = att_to_woman / total
        else:
            frac_man = frac_woman = 0.0

        results[head] = (frac_man, frac_woman)

    return results


# =========================
# MAIN
# =========================

def main():
    # ---- Load CSV & pick sentences ----
    df = pd.read_csv(CSV_PATH)
    df_clean = df[df["Question?"].notna()].reset_index(drop=True)

    row_a = df_clean.loc[IDX_A]
    row_b = df_clean.loc[IDX_B]

    is_corr_a = row_a["State of Result"] not in ERROR_STATES
    is_corr_b = row_b["State of Result"] not in ERROR_STATES

    sent_a = row_a["Question?"]
    sent_b = row_b["Question?"]

    print("=" * 80)
    print(f"Sentence A (idx {IDX_A}) | Correct? {is_corr_a}")
    print(sent_a)
    print("=" * 80)
    print(f"Sentence B (idx {IDX_B}) | Correct? {is_corr_b}")
    print(sent_b)
    print("=" * 80)

    # ---- Embeddings ----
    tokens_a, embs_a, pos_a, att_a = get_word_embeddings(sent_a)
    tokens_b, embs_b, pos_b, att_b = get_word_embeddings(sent_b)

    print("\nTokens for A:")
    print(list(enumerate(tokens_a)))
    print("\nTokens for B:")
    print(list(enumerate(tokens_b)))

    # ---- Intra-sentence cosines for his ↔ man/woman ----
    his_a = embs_a.get("his")
    man_a = embs_a.get("man")
    woman_a = embs_a.get("woman")

    his_b = embs_b.get("his")
    man_b = embs_b.get("man")
    woman_b = embs_b.get("woman")

    cos_his_man_a = cosine(his_a, man_a) if his_a is not None and man_a is not None else np.nan
    cos_his_woman_a = cosine(his_a, woman_a) if his_a is not None and woman_a is not None else np.nan

    cos_his_man_b = cosine(his_b, man_b) if his_b is not None and man_b is not None else np.nan
    cos_his_woman_b = cosine(his_b, woman_b) if his_b is not None and woman_b is not None else np.nan

    print("\nIntra-sentence cosines for Sentence A:")
    print("  cos(his, man)   =", cos_his_man_a)
    print("  cos(his, woman) =", cos_his_woman_a)

    print("\nIntra-sentence cosines for Sentence B:")
    print("  cos(his, man)   =", cos_his_man_b)
    print("  cos(his, woman) =", cos_his_woman_b)

    # ---- Cross-sentence cosines ----
    print("\nCross-sentence cosine similarities (A vs B):")
    for w in ["man", "woman", "his", "jacket"]:
        if w in embs_a and w in embs_b:
            sim = cosine(embs_a[w], embs_b[w])
            print(f"  cos({w}_A, {w}_B) = {sim:.3f}")

    # ---- Sanity check ----
    if CONTROL_WORD in embs_a and CONTROL_WORD in embs_b:
        sim_jacket = cosine(embs_a[CONTROL_WORD], embs_b[CONTROL_WORD])
        print(f"\nSanity check: cos({CONTROL_WORD}_A, {CONTROL_WORD}_B) = {sim_jacket:.3f}")

    # ---- Attention patterns ----
    att_heads_a = get_pronoun_attention_to_subjects(tokens_a, att_a, pos_a)
    att_heads_b = get_pronoun_attention_to_subjects(tokens_b, att_b, pos_b)

    rows = []
    for head in sorted(set(att_heads_a.keys()) | set(att_heads_b.keys())):
        frac_man_a, frac_woman_a = att_heads_a.get(head, (0.0, 0.0))
        frac_man_b, frac_woman_b = att_heads_b.get(head, (0.0, 0.0))
        rows.append({
            "head": head,
            "A_frac_man": frac_man_a,
            "A_frac_woman": frac_woman_a,
            "B_frac_man": frac_man_b,
            "B_frac_woman": frac_woman_b,
        })

    att_df = pd.DataFrame(rows)
    att_df["diff_man"] = att_df["B_frac_man"] - att_df["A_frac_man"]
    att_df["diff_woman"] = att_df["B_frac_woman"] - att_df["A_frac_woman"]
    att_df["abs_diff_total"] = att_df["diff_man"].abs() + att_df["diff_woman"].abs()

    # ------------------ VISUALIZATIONS ------------------

    # 1) Pronoun→subject cosine similarities bar chart
    labels = ["his→man", "his→woman"]
    A_vals = [cos_his_man_a, cos_his_woman_a]
    B_vals = [cos_his_man_b, cos_his_woman_b]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, A_vals, width, label=f"A (idx {IDX_A})")
    plt.bar(x + width/2, B_vals, width, label=f"B (idx {IDX_B})")
    plt.xticks(x, labels)
    plt.ylabel("Cosine similarity")
    plt.title("Pronoun→Subject Cosine Similarities (his)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Attention per head – plot top N heads by abs_diff_total
    top_n = 8
    important = att_df.sort_values("abs_diff_total", ascending=False).head(top_n)
    heads = important["head"].to_numpy()

    # Man attention
    A_man = important["A_frac_man"].to_numpy()
    B_man = important["B_frac_man"].to_numpy()

    x = np.arange(len(heads))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, A_man, width, label="A: pronoun→man")
    plt.bar(x + width/2, B_man, width, label="B: pronoun→man")
    plt.xticks(x, heads)
    plt.xlabel("Head")
    plt.ylabel("Attention fraction")
    plt.title("Pronoun→man attention (top heads by change)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Woman attention
    A_woman = important["A_frac_woman"].to_numpy()
    B_woman = important["B_frac_woman"].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, A_woman, width, label="A: pronoun→woman")
    plt.bar(x + width/2, B_woman, width, label="B: pronoun→woman")
    plt.xticks(x, heads)
    plt.xlabel("Head")
    plt.ylabel("Attention fraction")
    plt.title("Pronoun→woman attention (top heads by change)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
