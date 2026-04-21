import torch
import csv
import statistics
from transformers import T5Tokenizer, T5EncoderModel
import matplotlib.pyplot as plt

print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5EncoderModel.from_pretrained("google/flan-t5-large")
embedding_matrix = model.shared.weight.detach()

def clean_token(word):
    tokens = tokenizer.tokenize(word)
    if len(tokens) != 1:
        return None
    if tokens[0] == "▁":
        return None
    return tokens[0]

def cosine_similarity(tok1, tok2):
    id1 = tokenizer.convert_tokens_to_ids(tok1)
    id2 = tokenizer.convert_tokens_to_ids(tok2)
    v1 = embedding_matrix[id1]
    v2 = embedding_matrix[id2]
    return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()

def run_tests(category_name, pairs):
    results = []
    for w1, w2 in pairs:
        tok1 = clean_token(w1)
        tok2 = clean_token(w2)
        if tok1 is None or tok2 is None:
            print(f"SKIP {w1}/{w2} (bad tokenization)")
            continue
        cos = cosine_similarity(tok1, tok2)
        results.append(cos)
        print(f"{category_name:10s} {w1:12s} {w2:12s} cos={cos:.4f}")
    return results

# ------------------------------------------------
# WORD LISTS
# ------------------------------------------------

semantic_pairs = [
    ("doctor", "nurse"), ("king", "queen"), ("teacher", "student"), 
    ("anger", "rage"), ("car", "vehicle"), ("child", "kid"),
    ("aunt", "uncle"), ("happy", "joyful"), ("quick", "fast"),
    ("large", "big"), ("strong", "powerful"), ("brave", "courageous")
]

spelling_pairs = [
    ("man", "pan"), ("man", "ban"), ("pan", "fan"), ("cat", "cap"),
    ("cat", "cam"), ("bag", "bug"), ("bag", "big"), ("dog", "fog"),
    ("rock", "sock"), ("pin", "pan"), ("plan", "plant"), ("form", "from")
]

rhyming_pairs = [
    ("bear", "care"), ("hair", "stare"), ("write", "kite"),
    ("night", "fight"), ("blue", "shoe"), ("though", "throw"),
    ("loop", "soup"), ("read", "seed"), ("boat", "coat"), ("funny", "sunny")
]

control_pairs = [
    ("man", "gem"), ("cat", "vehicle"), ("sun", "computer"),
    ("dog", "storm"), ("bag", "charger"), ("moon", "chair"),
    ("river", "camera"), ("book", "pillow"), ("grass", "justice")
]

# ------------------------------------------------
# RUN EXPERIMENT
# ------------------------------------------------

category_map = {
    "semantic": semantic_pairs,
    "spelling": spelling_pairs,
    "rhyming": rhyming_pairs,
    "control": control_pairs,
}

all_results = {}

for cat, pairs in category_map.items():
    print(f"\n=== {cat.upper()} ===")
    scores = run_tests(cat, pairs)
    all_results[cat] = scores
    if len(scores) > 0:
        print(f"{cat} average = {statistics.mean(scores):.4f}, std = {statistics.pstdev(scores):.4f}")
    else:
        print("No valid data.")

# ------------------------------------------------
# SAVE TO CSV
# ------------------------------------------------
with open("phase2_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "word1", "word2", "similarity"])
    for cat, pairs in category_map.items():
        for (w1, w2), cos in zip(pairs, all_results[cat]):
            writer.writerow([cat, w1, w2, cos])

print("Saved results to phase2_results.csv")

# ------------------------------------------------
# BOXPLOT
# ------------------------------------------------

plt.figure(figsize=(10, 6))
data = [all_results["semantic"], all_results["spelling"], 
        all_results["rhyming"], all_results["control"]]

plt.boxplot(data, labels=["Semantic", "Spelling", "Rhyming", "Control"])
plt.title("Cosine similarity distribution by category")
plt.ylabel("Cosine similarity")
plt.grid(True, alpha=0.3)
plt.savefig("phase2_boxplot.png")
print("Saved boxplot to phase2_boxplot.png")
plt.show()
