from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.widgets import TextBox
from utils import ModelManager
from utils import is_int, is_float
from config import EMBED_CONFIGS

# mode: can be 0, 1, or 2, determines the type of visualization
mode = None
# tokens1: the input tokens
tokens1 = None
# encoder_hidden_states: contains the hidden states after running model.generate
encoder_hidden_states = None
# seq_len: length of input sequence
seq_len = 0
# num_layers: how many hidden states there are
num_layers = 0

# for embedding matrix visualizations
fig, axs = None, None
im1, im2 = None, None
cb1, cb2 = None, None
embedding1_layer_ax, embedding2_layer_ax, embedding1_layer, embedding2_layer = None, None, None, None
token_num_label_ax, token_num_label = None, None
emb1_layer_idx, emb2_layer_idx = 0, 0
token_idx = 0
min_embedding1_label_ax, min_embedding1_label, max_embedding1_label_ax, max_embedding1_label = None, None, None, None
min_embedding2_label_ax, min_embedding2_label, max_embedding2_label_ax, max_embedding2_label = None, None, None, None
max_embedding1, max_embedding2 = -float("inf"), -float("inf")
min_embedding1, min_embedding2 = float("inf"), float("inf")

# for cosine sim matrix visualization
ax = None

def plot_embedding(plot_idx, layer_idx):
    global emb1_layer_idx, emb2_layer_idx
    if plot_idx == 0:
        emb1_layer_idx = layer_idx
    if plot_idx == 1:
        emb2_layer_idx = layer_idx
    matrix_embedding_visualizations()

def submit_emb2_idx(text):
    if not text.isdigit() or int(text) > 23:
        print("Error: You entered an invalid layer number. Program exit")
        exit(1)
    cur_layer_idx = int(text)
    plot_embedding(1, cur_layer_idx)

def submit_emb1_idx(text):
    if not text.isdigit() or int(text) > 23:
        print("Error: You entered an invalid layer number. Program exit")
        exit(1)
    cur_layer_idx = int(text)
    plot_embedding(0, cur_layer_idx)

def submit_token_num_matrix(text):
    global token_idx
    if not text.isdigit() or int(text) > seq_len-1:
        print("Error: You entered an invalid token number. Program exit")
        exit(1)
    cur_token_idx = int(text)
    token_idx = cur_token_idx
    matrix_embedding_visualizations()

def submit_emb1_min(text):
    global min_embedding1, im1
    if not is_float(text):
        print("Error: You entered an invalid range for min. Program exit")
        exit(1)
    min_embedding1 = round(float(text), 2)
    im1.set_clim(min_embedding1, max_embedding1)

def submit_emb1_max(text):
    global max_embedding1, im1
    if not is_float(text):
        print("Error: You entered an invalid range for max. Program exit")
        exit(1)
    max_embedding1 = round(float(text), 2)
    im1.set_clim(min_embedding1, max_embedding1)

def submit_emb2_min(text):
    global min_embedding2, im2
    if not is_float(text):
        print("Error: You entered an invalid range for min. Program exit")
        exit(1)
    min_embedding2 = round(float(text), 2)
    im2.set_clim(min_embedding2, max_embedding2)

def submit_emb2_max(text):
    global max_embedding2, im2
    if not is_float(text):
        print("Error: You entered an invalid range for max. Program exit")
        exit(1)
    max_embedding2 = round(float(text), 2)
    im2.set_clim(min_embedding2, max_embedding2)

def matrix_embedding_visualizations():
    global fig, axs
    global embedding1_layer_ax, embedding2_layer_ax, embedding1_layer, embedding2_layer
    global im1, im2
    global cb1, cb2
    global token_num_label_ax, token_num_label
    global min_embedding1_label_ax, min_embedding1_label, max_embedding1_label_ax, max_embedding1_label
    global min_embedding2_label_ax, min_embedding2_label, max_embedding2_label_ax, max_embedding2_label
    global max_embedding1, max_embedding2, min_embedding1, min_embedding2

    if cb1 is not None: cb1.remove()
    if cb2 is not None: cb2.remove()

    # default token index for now:
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(50, 8))
        axs[0].set_title("Embedding 1 Plot")
        axs[1].set_title("Embedding 2 Plot")

        embedding1_layer_ax = fig.add_axes([0.30, 0.87, 0.05, 0.05])
        embedding2_layer_ax = fig.add_axes([0.70, 0.87, 0.05, 0.05])
        embedding1_layer = TextBox(embedding1_layer_ax, label='Layer ', initial="0")
        embedding2_layer = TextBox(embedding2_layer_ax, label='Layer ', initial="0")
        embedding1_layer.label.set_fontsize(16)
        embedding1_layer.text_disp.set_fontsize(16)
        embedding2_layer.label.set_fontsize(16)
        embedding2_layer.text_disp.set_fontsize(16)
        embedding1_layer.on_submit(submit_emb1_idx)
        embedding2_layer.on_submit(submit_emb2_idx)

        token_num_label_ax = fig.add_axes([0.55, 0.03, 0.05, 0.05])
        token_num_label = TextBox(token_num_label_ax, label='Token Number ', initial="0")
        token_num_label.label.set_fontsize(16)
        token_num_label.text_disp.set_fontsize(16)
        token_num_label.on_submit(submit_token_num_matrix)

        min_embedding1_label_ax = fig.add_axes([0.20, 0.15, 0.05, 0.05])
        min_embedding1_label = TextBox(min_embedding1_label_ax, label='Min ', initial="0")
        max_embedding1_label_ax = fig.add_axes([0.30, 0.15, 0.05, 0.05])
        max_embedding1_label = TextBox(max_embedding1_label_ax, label='Max ', initial="0")
        min_embedding1_label.on_submit(submit_emb1_min)
        max_embedding1_label.on_submit(submit_emb1_max)

        min_embedding2_label_ax = fig.add_axes([0.60, 0.15, 0.05, 0.05])
        min_embedding2_label = TextBox(min_embedding2_label_ax, label='Min ', initial="0")
        max_embedding2_label_ax = fig.add_axes([0.70, 0.15, 0.05, 0.05])
        max_embedding2_label = TextBox(max_embedding2_label_ax, label='Max ', initial="0")
        min_embedding2_label.on_submit(submit_emb2_min)
        max_embedding2_label.on_submit(submit_emb2_max)

    # updating title
    fig.suptitle(f"Current Token: {tokens1[token_idx]}", fontsize=16)

    # initialize to original embeddings
    # here, the embeddings are extracted from the first position all the way to where the index of jacket is
    original_embedding1 = encoder_hidden_states[emb1_layer_idx][0, token_idx, :]
    reshaped_original_embedding1 = original_embedding1.numpy().reshape((32, 32))
    im1 = axs[0].imshow(reshaped_original_embedding1)
    cb1 = fig.colorbar(im1, ax=axs[0], shrink=0.7, pad=0.1)
    original_embedding2 = encoder_hidden_states[emb2_layer_idx][0, token_idx, :]
    reshaped_original_embedding2 = original_embedding2.numpy().reshape((32, 32))
    im2 = axs[1].imshow(reshaped_original_embedding2)
    cb2 = fig.colorbar(im2, ax=axs[1], shrink=0.7, pad=0.1)

    # set the initial values for min and max labels
    min_embedding1, max_embedding1 = im1.get_clim()
    min_embedding2, max_embedding2 = im2.get_clim()
    min_embedding1_label.set_val(str(round(min_embedding1, 2)))
    max_embedding1_label.set_val(str(round(max_embedding1, 2)))
    min_embedding2_label.set_val(str(round(min_embedding2, 2)))
    max_embedding2_label.set_val(str(round(max_embedding2, 2)))

    plt.show()

def cosine_sim_lineplot():
    all_sims = []
    for cur_token in range(seq_len):
        original_embedding = encoder_hidden_states[0][0, cur_token, :]
        cosine_similarities = []
        for layer in encoder_hidden_states:
            cur_embedding = layer[0, cur_token, :]
            # print(f"Current embedding: {cur_embedding}")
            sim = F.cosine_similarity(original_embedding, cur_embedding, dim=0).item()
            cosine_similarities.append(sim)
        all_sims.append(cosine_similarities)

    # print(outputs[0][1:-1])
    # output_text = tokenizer.decode(outputs[0][0])
    # print(f"output: {output_text}")

    for idx, cos_sim in enumerate(all_sims):
        plt.plot(cos_sim, marker='o', label=tokens1[idx])
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity to input embedding")
    plt.legend()
    plt.show()

def submit_token_num_cos(text):
    global token_idx
    if not text.isdigit() or int(text) > seq_len-1:
        print("Error: You entered an invalid token number. Program exit")
        exit(1)
    cur_token_idx = int(text)
    token_idx = cur_token_idx
    matrix_cosine_sim_visualization()

def matrix_cosine_sim_visualization():
    global cb1, fig, ax, token_num_label_ax, token_num_label
    global token_idx
    all_sims = []

    if cb1 is not None: cb1.remove()

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(50, 8))

        token_num_label_ax = fig.add_axes([0.55, 0.03, 0.05, 0.05])
        token_num_label = TextBox(token_num_label_ax, label='Token Number ', initial="0")
        token_num_label.label.set_fontsize(16)
        token_num_label.text_disp.set_fontsize(16)
        token_num_label.on_submit(submit_token_num_cos)

    # updating title
    fig.suptitle(f"Current Token: {tokens1[token_idx]}", fontsize=16)

    for layer_i in encoder_hidden_states:
        layer_i_embedding = layer_i[0, token_idx, :]
        layer_ij_sims = []
        for layer_j in encoder_hidden_states:
            layer_j_embedding = layer_j[0, token_idx, :]
            sim = F.cosine_similarity(layer_i_embedding, layer_j_embedding, dim=0).item()
            layer_ij_sims.append(sim)
        all_sims.append(layer_ij_sims)
    all_sims_np = np.array(all_sims)
    im = ax.imshow(all_sims_np)
    cb1 = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.1)

    plt.show()

def print_usage():
    print("Usage: python extract_embeddings.py <input prompt> -mode <mode number>")
    print(
        "<mode_number>: supports 0, 1, and 2")
    print(
        "      0: Cosine similarity between embedding at Layer 0 and every other layer is plotted via line graph.")
    print(
        "      1: Embeddings are represented as 32x32 matrices, can access embeddings at every layer with text boxes. \n"
        "         Intended for exploration/comparison.")
    print("      2: Cosine similarity between every layer and every other layer is plotted via a 24x24 matrix")
    print("Example Usage: \n"
          "python extract_embeddings.py -mode 1"
          "\n\n")

def main():
    global mode
    global tokens1, outputs, encoder_hidden_states, seq_len, num_layers
    if len(sys.argv) < 3:
        print_usage()
        exit(1)
    elif len(sys.argv) == 3:
        flag = sys.argv[1]
        if flag == "-mode":
            if not is_int(sys.argv[2]):
                print_usage()
                exit(1)
            else:
                mode = int(sys.argv[2])
        else:
            print_usage()
            exit(1)
    if mode < 0 or mode > 2:
        print_usage()
        exit(1)

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

    print(EMBED_CONFIGS['introduction'])

    print(EMBED_CONFIGS[f'description-details-{mode}'])
    print('\n')


    user_input = input("Please enter your prompt here: ")
    #getting the sentence from the user
    inputs = tokenizer(user_input, return_tensors="pt")
    inputs_ids = inputs.input_ids

    #passing the sentence in to get the hidden states
    outputs = model.generate(
        input_ids=inputs_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_length=20
    )

    #convert the index ids to tokens
    tokens1 = tokenizer.convert_ids_to_tokens(inputs_ids[0])
    print(f"Input IDS: {inputs_ids[0]}")
    print(f"Tokens: {tokens1}")
    print(f"Model output: {tokenizer.decode(outputs[0][0])}")

    #extracting encoder hidden states
    encoder_hidden_states = outputs.encoder_hidden_states

    seq_len = encoder_hidden_states[0].shape[1]
    num_layers = len(encoder_hidden_states)

    if mode == 0:
        cosine_sim_lineplot()
    elif mode == 1:
        matrix_embedding_visualizations()
    elif mode == 2:
        matrix_cosine_sim_visualization()

if __name__ == '__main__':
    main()