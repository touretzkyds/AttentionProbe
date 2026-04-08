import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load T5 model ---
model_name = "google/flan-t5-large"  # or "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
model.eval()

# --- Sentences ---
sentences = [
    "The man showed the woman his jacket.",
    "The man showed the woman her jacket."
]

# --- Pronoun and noun indices (you can adjust after tokenization) ---
pronouns_dict = {'his': None, 'her': None}  # will fill automatically
nouns_dict = {'man': None, 'woman': None}

# --- Function to analyze ---
def analyze_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Auto-find indices
    pron_idx = [i for i, t in enumerate(tokens) if t in ['▁his', '▁her']]
    noun_idx = [i for i, t in enumerate(tokens) if t in ['▁man', '▁woman']]

    if len(pron_idx) == 0 or len(noun_idx) == 0:
        print("Warning: Could not find pronoun or noun in tokens:", tokens)
    
    outputs = model.encoder(input_ids=inputs['input_ids'], output_attentions=True, output_hidden_states=True)
    
    hidden_states = torch.stack(outputs.hidden_states)  # layers x batch x seq_len x hidden
    attentions = torch.stack(outputs.attentions)        # layers x batch x heads x seq_len x seq_len
    
    # --- Cosine similarity of pronoun to nouns across layers ---
    pronoun_sim = []
    for layer in range(hidden_states.shape[0]):
        pron_vec = hidden_states[layer, 0, pron_idx, :].mean(0)  # average if multiple pronouns
        noun_vecs = hidden_states[layer, 0, noun_idx, :]
        sims = torch.nn.functional.cosine_similarity(pron_vec.unsqueeze(0), noun_vecs)
        pronoun_sim.append(sims.detach().numpy())
    pronoun_sim = torch.tensor(pronoun_sim).numpy()
    
    # --- Attention from pronoun to nouns ---
    pronoun_attn = []
    for layer in range(attentions.shape[0]):
        layer_attn = attentions[layer][0]  # heads x seq_len x seq_len
        attn_to_nouns = layer_attn[:, pron_idx, :][:, :, noun_idx]  # heads x pronouns x nouns
        attn_to_nouns = attn_to_nouns.mean(1)  # average over pronouns
        pronoun_attn.append(attn_to_nouns.detach().numpy())
    pronoun_attn = torch.tensor(pronoun_attn).numpy()  # layers x heads x nouns
    
    return tokens, pronoun_sim, pronoun_attn

# --- Run ---
for sentence in sentences:
    tokens, sim, attn = analyze_sentence(sentence)
    print(f"\nSentence: {sentence}")
    print(f"Tokens: {tokens}")
    print("Hidden-state cosine similarity (pronoun -> nouns) per layer:\n", sim)
    print("Attention from pronoun to nouns (layers x heads x nouns):\n", attn)
    
    # Optional visualization
    plt.figure(figsize=(8,5))
    sns.heatmap(sim, xticklabels=['man','woman'], yticklabels=range(sim.shape[0]), cmap='viridis')
    plt.xlabel('Nouns')
    plt.ylabel('Layers')
    plt.title(f'Pronoun-Noun Cosine Similarity: "{sentence}"')
    plt.show()

    plt.figure(figsize=(8,5))
    sns.heatmap(attn[:,0,:], xticklabels=['man','woman'], yticklabels=range(attn.shape[0]), cmap='magma')
    plt.xlabel('Nouns')
    plt.ylabel('Layers')
    plt.title(f'Pronoun-Noun Attention, Head 0: "{sentence}"')
    plt.show()
