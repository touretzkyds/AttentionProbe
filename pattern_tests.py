import torch
from transformers import T5Tokenizer, T5EncoderModel

# ---- LOAD MODEL ----
print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5EncoderModel.from_pretrained("google/flan-t5-base")

print("Detaching embedding matrix...")
matrix = model.shared.weight.detach()

# ---- HELPERS ----
def safe_similarity(word1, word2, tokenizer, matrix):
    t1 = tokenizer.tokenize(word1)
    t2 = tokenizer.tokenize(word2)

    if len(t1) != 1 or len(t2) != 1:
        print(f"SKIP: '{word1}'->{t1}, '{word2}'->{t2}")
        return None

    id1 = tokenizer.convert_tokens_to_ids(t1)[0]
    id2 = tokenizer.convert_tokens_to_ids(t2)[0]

    vec1 = matrix[id1]
    vec2 = matrix[id2]

    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()


def run_group(name, pairs, tokenizer, matrix):
    print(f"\n=== {name} ===")
    scores = []

    for w1, w2 in pairs:
        cos = safe_similarity(w1, w2, tokenizer, matrix)
        if cos is not None:
            print(f"{w1:10s}  {w2:10s}  cos={cos:.4f}")
            scores.append(cos)

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n{name} AVERAGE: {avg:.4f}")
    else:
        print("\nNO VALID PAIRS\n")


# ---- EXPERIMENT GROUPS ----
semantic_pairs = [
    ("lead", "led"),
    ("read", "red"),
    ("sleep", "slept"),
    ("king", "queen"),
    ("doctor", "nurse"),
    ("teacher", "student"),
]

spelling_pairs = [
    ("man", "pan"), ("man", "ban"), ("man", "fan"),
    ("pan", "fan"), ("pan", "can"), ("pan", "man"),

    ("cat", "cap"), ("cat", "cam"), ("cat", "cup"),
    ("bag", "bug"), ("bag", "big"),
    ("dog", "fog"), ("dog", "log"),
]

rhyming_pairs = [
    ("knight", "night"),
    ("light", "night"),
    ("light", "fight"),
    ("ball", "call"),
    ("cool", "pool"),
    ("pool", "tool"),
    ("flower", "flour"),
]

control_pairs = [
    ("man", "gem"),
    ("there", "gem"),
    ("cat", "vehicle"),
    ("sun", "computer"),
    ("dog", "storm"),
    ("bag", "charger"),
]

# ---- RUN ALL GROUPS ----
run_group("Semantic similarity", semantic_pairs, tokenizer, matrix)
run_group("Spelling similarity", spelling_pairs, tokenizer, matrix)
run_group("Rhyming similarity", rhyming_pairs, tokenizer, matrix)
run_group("Control unrelated", control_pairs, tokenizer, matrix)
