"""
Pair-wise comparison script for pronoun resolution.

Given two sentence indices (in the cleaned CSV),
this script compares:

- Pronoun embeddings across sentences
- Subject embeddings across sentences
- Pronoun↔subject cosine similarities inside each sentence
- Attention from pronoun -> man / woman for each head
- A control word ("jacket") across sentences

Adjust IDX_A and IDX_B to choose which sentences to compare.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

# =========================
# CONFIG
# =========================

CSV_PATH = "/home/navya/Attention Probe Results - Sheet1.csv"
MODEL_NAME = "google/flan-t5-large"

# Choose the two sentence row indices (after cleaning)
IDX_A = 18   # e.g. "wrong" sentence
IDX_B = 28   # e.g. "right" sentence

CONTROL_WORD = "jacket"
TARGET_WORDS = ["man", "woman", "his", "her", CONTROL_WORD]

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
    last_hidden = out.last_hidden_state[0]       # [seq_len, hidden_dim]
    attentions = out.attentions                  # list[layers] of [1, heads, seq, seq]
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
    using the last layer.
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

    # correctness flags (if you care)
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

    # ---- Embeddings for both sentences ----
    tokens_a, embs_a, pos_a, att_a = get_word_embeddings(sent_a)
    tokens_b, embs_b, pos_b, att_b = get_word_embeddings(sent_b)

    # Show tokenization (helpful sanity)
    print("\nTokens for A:")
    print(list(enumerate(tokens_a)))
    print("\nTokens for B:")
    print(list(enumerate(tokens_b)))

    # ---- Pronoun↔subject cosine similarities inside each sentence ----
    def intra_sentence_cosines(label, embs, correct_is_man=True):
        h = embs.get("his")
        her = embs.get("her")
        man = embs.get("man")
        woman = embs.get("woman")

        print(f"\nIntra-sentence cosines for {label}:")
        if h is not None and man is not None:
            print("  cos(his, man)   =", cosine(h, man))
        if h is not None and woman is not None:
            print("  cos(his, woman) =", cosine(h, woman))
        if her is not None and man is not None:
            print("  cos(her, man)   =", cosine(her, man))
        if her is not None and woman is not None:
            print("  cos(her, woman) =", cosine(her, woman))

    intra_sentence_cosines("Sentence A", embs_a)
    intra_sentence_cosines("Sentence B", embs_b)

    # ---- Cross-sentence comparisons of embeddings ----
    print("\nCross-sentence cosine similarities (A vs B):")
    for w in TARGET_WORDS:
        if w in embs_a and w in embs_b:
            sim = cosine(embs_a[w], embs_b[w])
            print(f"  cos({w}_A, {w}_B) = {sim:.3f}")

    # ---- Sanity check: control word ----
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
        # we can define "correct" subject per sentence manually in the writeup,
        # but here we just show both.
        rows.append({
            "head": head,
            "A_frac_man": frac_man_a,
            "A_frac_woman": frac_woman_a,
            "B_frac_man": frac_man_b,
            "B_frac_woman": frac_woman_b,
            "diff_man": frac_man_b - frac_man_a,
            "diff_woman": frac_woman_b - frac_woman_a,
        })

    att_df = pd.DataFrame(rows)
    print("\nPer-head attention from pronoun → subjects (A vs B):")
    print(att_df.head(20))

    # Identify heads with biggest change in attention to man/woman
    att_df["abs_diff_total"] = (att_df["diff_man"].abs() + att_df["diff_woman"].abs())
    important = att_df.sort_values("abs_diff_total", ascending=False).head(10)

    print("\nTop heads whose pronoun→subject attention changed most between A and B:")
    print(important[["head", "A_frac_man", "A_frac_woman",
                    "B_frac_man", "B_frac_woman", "abs_diff_total"]])

if __name__ == "__main__":
    main()
