#load imports
print("starting")
import torch
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5EncoderModel
from torch.nn.functional import cosine_similarity
print("imports done")

#loading all of the model stuff
model_name = "google/flan-t5-large"

tokenizer = T5Tokenizer.from_pretrained(model_name)
print("Tokenizer loaded")
model = T5EncoderModel.from_pretrained(
    model_name,
    output_attentions=True
)
print("Model loaded")

#types of heads that we believe are interesting
IMPORTANT_HEADS = sorted({1, 6, 8, 9, 11, 12, 14, 15})

#all types of pronouns that we could have in our sentences(doesn't mean that these are all types of pronouns)
PRONOUNS = {"he", "she", "her", "him", "his", "hers"}

#some helpers functions that we have
#tokenizes each part of the sentence 
def tokenize(sentence):
    enc = tokenizer(sentence, return_tensors="pt")#tokenizes the whole sentences and turns it into ids
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0]) #converts the ids back into readable tokens
    return enc, tokens #the dictionary for the pytorch tensors that we get from the sentence and the tokens 

#takes these tokenized versions and finds the indices in the embedding matrix
def find_pronoun_indices(tokens):
    idxs = []
    for i, tok in enumerate(tokens):
        clean = tok.replace("▁", "").lower()
        if clean in PRONOUNS:
            idxs.append(i)
    return idxs

#get the last layer attention heads
def get_last_layer_attentions(enc):#taking in the tokenized version of the sentence

    with torch.no_grad():#computes faster
        out = model(**enc) #out contains things like hidden states and encoder attention matrices, we care about the encoder attentions
    # out.attentions shape: [num_layers, batch, heads, seq, seq]
    last = out.attentions[-1][0]  # [heads, seq, seq] #gets the last layer, and removes batch dimension because we only input one sentence
    return last

#the first kind of layout
def plot_bar_layout(tokens, last_layer_att, pronoun_idx, heads=IMPORTANT_HEADS):
    seq_len = len(tokens)
    num_heads = len(heads)

    fig, axes = plt.subplots(num_heads, 1, figsize=(12, 2 * num_heads), sharex=True)
    if num_heads == 1:
        axes = [axes]

    token_positions = list(range(seq_len))
    x_labels = [t.replace("▁", "_") for t in tokens]

    for ax, head in zip(axes, heads):
        att_vec = last_layer_att[head, pronoun_idx].detach().cpu().numpy()

        ax.bar(token_positions, att_vec)
        ax.set_ylabel(f"H{head}", rotation=0, labelpad=25, fontsize=9)
        ax.set_ylim(0, max(att_vec) * 1.1 if att_vec.max() > 0 else 1.0)

        # highlight pronoun position with a vertical line
        ax.axvline(pronoun_idx, color="red", linestyle="--", alpha=0.6)

        # optionally only label a few x-ticks to avoid clutter
        ax.set_xticks(token_positions)
        ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=8)

    axes[0].set_title(f"Layout A: Attention from pronoun (idx {pronoun_idx}) to all tokens")
    plt.tight_layout()
    plt.show()

#second type of layout
def plot_spotlight_ring(tokens, att_scores, pronoun_idx):
    import matplotlib.pyplot as plt
    import numpy as np

    N = len(tokens)

    # Normalize attention scores for line lengths
    if max(att_scores) == 0:
        norm_scores = np.zeros_like(att_scores)
    else:
        norm_scores = att_scores / max(att_scores)

    # Adjust radius range
    MIN_R = 0.5     # minimum distance from center
    MAX_R = 2.5     # maximum distance from center

    # Compute radius for each token
    radii = MIN_R + norm_scores * (MAX_R - MIN_R)

    # Spread tokens evenly around a circle (angles)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot pronoun at center
    px, py = 0, 0
    ax.scatter(px, py, color='red', s=300, zorder=3)
    ax.text(px, py, tokens[pronoun_idx], ha='center', va='center', color='white', fontsize=12)

    # Plot each token
    for i, ang in enumerate(angles):
        if i == pronoun_idx:
            continue

        # Position determined by scaled radius
        r = radii[i]
        x = r * np.cos(ang)
        y = r * np.sin(ang)

        ax.scatter(x, y, color='black')
        ax.text(x, y, tokens[i], ha='center', va='center')

        # Draw connecting line whose length = radius
        ax.plot([px, x], [py, y], color='blue', linewidth=1)

        # Annotate with the numeric attention score
        mx, my = (px + x) / 2, (py + y) / 2
        ax.text(mx, my, f"{att_scores[i]:.2f}", fontsize=9)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Spotlight Ring: Line Length Reflects Attention Strength")
    plt.show()

#third type of layout
def plot_bar_head(tokens, att_scores, head_number, pronoun_token):
    import matplotlib.pyplot as plt
    import numpy as np

    sorted_pairs = sorted(
        list(enumerate(att_scores)),
        key=lambda x: x[1],
        reverse=True
    )

    sorted_tokens = [tokens[i] for i, s in sorted_pairs]
    sorted_scores = [s for i, s in sorted_pairs]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_tokens[::-1], sorted_scores[::-1], color='skyblue')
    plt.xlabel("Attention score")
    plt.title(f"Head {head_number} attention from pronoun '{pronoun_token}'")
    plt.tight_layout()
    plt.show()


#main loop

def main():
    #asks the user for a sentence
    sentence = input("Enter a sentence: ").strip()

    #gets the tokens of the sentence
    enc, tokens = tokenize(sentence)
    #gets the last layer attention heads 
    last_layer_att = get_last_layer_attentions(enc)

    #prints out each token
    print("\nTokens:")
    for i, tok in enumerate(tokens):
        print(f"{i:2d}: {tok}")

    #goes to find the pronouns based on the pronouns we want
    pronoun_idxs = find_pronoun_indices(tokens)

    #shows what pronouns were detected
    print("\nPronouns detected at indices:")
    for i in pronoun_idxs:
        print(f"  {i}: {tokens[i]}")

    # lets the user select pronoun
    while True:
        p_idx = int(input("\nChoose a pronoun index to visualize: "))
        if p_idx not in pronoun_idxs:
            print("Not a pronoun index. Try again.")
            continue
        break

    print("\nChoose a visualization layout(bars, ring, headbars):")
    layout = input("Enter choice (1,2,3): ").strip()

    # Layout 1 — existing bar layout (all heads)
    if layout == "1":
        plot_bar_layout(tokens, last_layer_att, p_idx)
        return

    # For layouts 2 & 3, user must pick a head
    print("\nAvailable heads:", IMPORTANT_HEADS)
    while True:
        try:
            head_choice = int(input("Choose a head: "))
        except ValueError:
            print("Enter an integer.")
            continue
        break

    att_scores = last_layer_att[head_choice, p_idx].detach().cpu().numpy() #extracts the attention values based on the attention head choice, the pronoun token
    #basically says for this head, how much attention is paid to every word
    #detach speeds up the processing
    #makes sure it's computed on cpu since i don't have a gpu
    #converts the tensors into a numpy array so we can plot

    # Layout 2 — spotlight ring
    if layout == "2"":
        plot_spotlight_ring(tokens, att_scores, p_idx)
        return

    # Layout 3 — bar chart per head
    if layout == "3" or layout.lower() == "headbars":
        plot_bar_head(tokens, att_scores, head_choice, tokens[p_idx])
        return

    print("Unknown choice. Defaulting to stacked bars.")
    plot_bar_layout(tokens, last_layer_att, p_idx)

if __name__ == "__main__":
    main()
