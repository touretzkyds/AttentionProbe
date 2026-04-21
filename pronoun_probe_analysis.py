"""
Full analysis script for pronoun resolution / attention probe.

What it does:
- Loads "Attention Probe Results - Sheet1.csv"
- Selects target rows (e.g., 17–20, 27–30)
- Extracts last-layer embeddings for man/woman/his/her/jacket
- Computes cosine similarities between pronouns and subjects
- Sanity-checks a constant word ("jacket") across sentences
- Extracts attention from pronoun → man/woman in the last encoder layer
- Produces Pandas DataFrames you can inspect or export


Adjust the CONFIG section as needed (model name, CSV path, row indices).
"""

import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, T5EncoderModel


CSV_PATH = "/home/navya/Attention Probe Results - Sheet1.csv"


TARGET_ROWS = [17, 18, 19, 20, 27, 28, 29, 30]

MODEL_NAME = "google/flan-t5-large" 

TARGET_WORDS = ["man", "woman", "his", "her", "jacket"]

# States that indicate the model got something wrong or weird
ERROR_STATES = {
    "Incorrect Answer",
    "Inconsistent with one or more runs",
    "Ambigious sentence",
    "Hallucination ",
    "Abnormality",
}


print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# AutoModel gives you the encoder for encoder-decoder models like T5
model = T5EncoderModel.from_pretrained(
    MODEL_NAME,
    output_attentions=True,
    output_hidden_states=True
)
model.eval()
print("Model loaded.")

def get_tokens(sentence):
    enc = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return enc, tokens


def find_word_indices(tokens, word):
    word = word.lower()
    idxs = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("▁", "").lower()
        if clean_tok == word:
            idxs.append(i)
    return idxs


def get_last_hidden_and_attentions(sentence):
    enc, tokens = get_tokens(sentence)
    with torch.no_grad():
        outputs = model(**enc)

    if hasattr(outputs, "encoder_last_hidden_state"):
        last_hidden = outputs.encoder_last_hidden_state[0]  
        if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions is not None:
            attentions = outputs.encoder_attentions  
        else:
            attentions = outputs.attentions  
    else:
        last_hidden = outputs.last_hidden_state[0] 
        attentions = outputs.attentions

    return tokens, last_hidden, attentions


def get_word_embeddings_from_sentence(sentence, target_words=TARGET_WORDS):
    tokens, last_hidden, _ = get_last_hidden_and_attentions(sentence)
    word_embs = {}

    for w in target_words:
        idxs = find_word_indices(tokens, w)
        if not idxs:
            continue
        vec = last_hidden[idxs].mean(dim=0)
        word_embs[w] = vec

    return tokens, word_embs


def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def get_pronoun_subject_attentions(sentence):
    tokens, _, attentions = get_last_hidden_and_attentions(sentence)

    last_layer_att = attentions[-1][0]  # [heads, seq_len, seq_len]

    his_idxs = find_word_indices(tokens, "his")
    her_idxs = find_word_indices(tokens, "her")
    pronoun_idxs = his_idxs + her_idxs

    man_idxs = find_word_indices(tokens, "man")
    woman_idxs = find_word_indices(tokens, "woman")

    rows = []

    if not pronoun_idxs or (not man_idxs and not woman_idxs):
        return tokens, rows

    num_heads = last_layer_att.shape[0]
    seq_len = last_layer_att.shape[1]

    for head in range(num_heads):
        att_to_man = 0.0
        att_to_woman = 0.0
        total_att = 0.0

        for p in pronoun_idxs:
            att_vec = last_layer_att[head, p] 
            total_att += float(att_vec.sum().item())

            if man_idxs:
                att_to_man += float(att_vec[man_idxs].sum().item())
            if woman_idxs:
                att_to_woman += float(att_vec[woman_idxs].sum().item())

        if total_att > 0:
            frac_man = att_to_man / total_att
            frac_woman = att_to_woman / total_att
        else:
            frac_man = 0.0
            frac_woman = 0.0

        rows.append({
            "head": head,
            "pronoun_positions": pronoun_idxs,
            "man_positions": man_idxs,
            "woman_positions": woman_idxs,
            "att_to_man": att_to_man,
            "att_to_woman": att_to_woman,
            "frac_man": frac_man,
            "frac_woman": frac_woman,
        })

    return tokens, rows

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    df_clean = df[df["Question?"].notna()].reset_index(drop=True)

    print("Total rows with questions:", len(df_clean))
    print("Using target rows:", TARGET_ROWS)

    subset = df_clean.loc[TARGET_ROWS].copy()

    subset["is_correct"] = ~subset["State of Result"].isin(ERROR_STATES)

    print("\nSelected sentences:")
    for i, row in subset.iterrows():
        print("=" * 80)
        print(f"Row index: {i}")
        print("Correct?" , row["is_correct"])
        print("Question:", row["Question?"])

    print("\nComputing embeddings and cosine similarities...")
    emb_rows = []

    jacket_embs = {}

    for i, row in subset.iterrows():
        sentence = row["Question?"]
        is_correct = row["is_correct"]

        tokens, word_embs = get_word_embeddings_from_sentence(sentence)

        rec = {
            "row_index": i,
            "sentence": sentence,
            "is_correct": is_correct,
        }

        if "his" in word_embs and "man" in word_embs:
            rec["cos_his_man"] = cosine(word_embs["his"], word_embs["man"])
        else:
            rec["cos_his_man"] = np.nan

        if "his" in word_embs and "woman" in word_embs:
            rec["cos_his_woman"] = cosine(word_embs["his"], word_embs["woman"])
        else:
            rec["cos_his_woman"] = np.nan

        if "her" in word_embs and "man" in word_embs:
            rec["cos_her_man"] = cosine(word_embs["her"], word_embs["man"])
        else:
            rec["cos_her_man"] = np.nan

        if "her" in word_embs and "woman" in word_embs:
            rec["cos_her_woman"] = cosine(word_embs["her"], word_embs["woman"])
        else:
            rec["cos_her_woman"] = np.nan

        if "jacket" in word_embs:
            jacket_embs[i] = word_embs["jacket"]

        emb_rows.append(rec)

    emb_df = pd.DataFrame(emb_rows)
    print("\nPronoun–subject cosine similarity table:")
    print(emb_df[[
        "row_index", "is_correct",
        "cos_his_man", "cos_his_woman",
        "cos_her_man", "cos_her_woman"
    ]])

    print("\nComputing sanity-check similarities for 'jacket'...")
    jacket_rows = []
    for (i1, v1), (i2, v2) in itertools.combinations(jacket_embs.items(), 2):
        sim = cosine(v1, v2)
        jacket_rows.append({
            "row_index_1": i1,
            "row_index_2": i2,
            "cos_jacket_jacket": sim,
        })

    jacket_df = pd.DataFrame(jacket_rows)
    print("\n'jacket' embedding similarities:")
    print(jacket_df)

    print("\nComputing attention from pronoun → man/woman in last layer...")
    att_all_rows = []

    for i, row in subset.iterrows():
        sentence = row["Question?"]
        is_correct = row["is_correct"]

        tokens, att_rows = get_pronoun_subject_attentions(sentence)

        for att_rec in att_rows:
            rec = {
                "row_index": i,
                "sentence": sentence,
                "is_correct": is_correct,
                "head": att_rec["head"],
                "att_to_man": att_rec["att_to_man"],
                "att_to_woman": att_rec["att_to_woman"],
                "frac_man": att_rec["frac_man"],
                "frac_woman": att_rec["frac_woman"],
                "man_positions": att_rec["man_positions"],
                "woman_positions": att_rec["woman_positions"],
                "pronoun_positions": att_rec["pronoun_positions"],
            }
            att_all_rows.append(rec)

    att_df = pd.DataFrame(att_all_rows)
    print("\nSample of attention DataFrame:")
    print(att_df.head())

    if not att_df.empty:
        print("\nAverage frac_man / frac_woman by correctness:")
        summary = att_df.groupby("is_correct")[["frac_man", "frac_woman"]].mean()
        print(summary)

        # ---- SAVE RESULTS ----
    print("\nSaving results...")

    emb_df.to_csv("cosine_results_no_correcteness.csv", index=False)
    jacket_df.to_csv("jacket_sanity_no_correctness.csv", index=False)
    att_df.to_csv("attention_results_no_correctness.csv", index=False)

    print("Saved:")
    print(" - cosine_results_no_correctness.csv")
    print(" - jacket_sanity_no_correctness.csv")
    print(" - attention_results_no_correctness.csv")

if __name__ == "__main__":
    main()
