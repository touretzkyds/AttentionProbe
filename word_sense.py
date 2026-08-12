"""
Ambiguous Word Sense Visualizer
================================
Layer-wise probing of FLAN-T5 encoder representations across lexical ambiguity contexts (including Layer 0 input embeddings).
"""

import os
import json
import re
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from openai import OpenAI

# CONFIG
OPENAI_MODEL = "gpt-5.5"
T5_MODEL_NAME = "google/flan-t5-large"
MODEL_DISPLAY_NAME = T5_MODEL_NAME.split("/")[-1].upper()

st.set_page_config(page_title="Ambiguous Word Sense Visualizer", layout="wide")


# 1. JSON Schema & System Prompt (Enforcing POS, Sanity & Coarse Sense Separation)

AMBIGUITY_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ambiguous_word_senses",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_valid": {
                    "type": "boolean",
                    "description": "True if the word naturally exists and is commonly used in English as the requested POS, False otherwise."
                },
                "error_message": {
                    "type": "string",
                    "description": "Explanation if is_valid is False. Leave empty if is_valid is True."
                },
                "word": {"type": "string"},
                "pos": {
                    "type": "string",
                    "enum": ["noun", "verb", "adjective", "adverb"],
                    "description": "Part of speech requested"
                },
                "sense1_word": {
                    "type": "string",
                    "pattern": "^[a-zA-Z]*$",
                    "description": "Single-word synonym for primary sense 1 (e.g., 'uncommon'). Empty if is_valid is False."
                },
                "sense2_word": {
                    "type": "string",
                    "pattern": "^[a-zA-Z]*$",
                    "description": "Single-word synonym for coarse-grained sense 2 belonging to a completely different domain (e.g., 'raw' or 'undercooked'). Empty if is_valid is False."
                },
                "c1_sent": {
                    "type": "string",
                    "description": "Short sentence (5-8 words) using the target word in sense 1 (leave empty if is_valid is False)."
                },
                "c2_sent": {
                    "type": "string",
                    "description": "Minimal pair sentence (5-8 words) using the target word in sense 2 (leave empty if is_valid is False)."
                }
            },
            "required": ["is_valid", "error_message", "word", "pos", "sense1_word", "sense2_word", "c1_sent", "c2_sent"],
            "additionalProperties": False
        }
    }
}

SYSTEM_PROMPT = """You are a precise linguistics data generator for a lexical-ambiguity probing experiment.

CRITICAL VALIDATION STEP:
1. First, check if the given word natively and naturally exists in standard English usage as the requested Part of Speech (POS).
2. If the word DOES NOT exist or is extremely forced/uncommon/nonsense as the requested POS (e.g., asking for "mouse" as an adverb or "apple" as a verb), set `is_valid: false` and provide a clear explanation in `error_message`. Leave all sentence and synonym fields as empty strings.
3. Do NOT invent obscure senses, artificial jargon, or ungrammatical sentences to satisfy an invalid request.

CRITICAL SENSE SEPARATION RULE (COARSE-GRAINED POLYSEMY / HOMONYMY):
- The two senses MUST BE SIGNIFICANTLY AND DRAMATICALLY DIFFERENT in meaning and domain.
- Do NOT output near-synonyms, nuanced variations, or fine-grained sub-senses of the same concept (e.g., do NOT pair 'uncommon' with 'scarce' or 'infrequent' for the word 'rare').
- Focus on coarse-grained polysemy or homonymy across entirely separate semantic domains.
  - Example for 'rare' (adjective): Sense 1 = 'uncommon' / 'scarce' (frequency); Sense 2 = 'undercooked' / 'raw' (meat cooking level).
  - Example for 'bank' (noun): Sense 1 = 'riverbank' / 'shore' (geography); Sense 2 = 'vault' / 'financial' (finance).
  - Example for 'bark' (verb): Sense 1 = 'howl' (dog sound); Sense 2 = 'shout' / 'yell' (human voice action).

IF IS_VALID IS TRUE:
- sense1_word: A single-word common English synonym/hypernym representing Domain A.
- sense2_word: A single-word common English synonym/hypernym representing a completely distinct Domain B.
- c1_sent: A short (5-8 words), natural, grammatically correct sentence using the target word in Sense 1.
- c2_sent: A short (5-8 words), natural, grammatically correct sentence using the target word in Sense 2.

CRITICAL FORMAT RULES:
- BOTH sense1_word and sense2_word MUST be strictly single words (alphabetic characters only).
- Target word must appear verbatim in both c1_sent and c2_sent.
- Ensure total semantic sanity: context sentences must make natural sense to a native English speaker.
"""


def extract_sense_word(sentence: str, target_word: str, default_word: str) -> str:
    """Dynamically extracts the sense synonym word from an edited reference sentence."""
    words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
    stopwords = {"a", "an", "the", "to", "means", "is", "being", "are", target_word.lower()}
    filtered = [w for w in words if w not in stopwords]
    if filtered:
        return filtered[-1]
    return default_word.lower()


def generate_contexts(word: str, pos: str, max_retries: int = 3) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY environment variable not found!")
        st.stop()
        
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                response_format=AMBIGUITY_RESPONSE_SCHEMA,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Ambiguous word: {word}\nRequired POS: {pos}"},
                ],
            )
            raw = json.loads(resp.choices[0].message.content)
            
            # POS Sanity Check
            if not raw.get("is_valid", True):
                error_msg = raw.get(
                    "error_message", 
                    f"The word '{word}' does not exist as a {pos.upper()} in standard English."
                )
                raise ValueError(error_msg)

            target_word = raw["word"].strip()
            s1_word = raw["sense1_word"].strip().lower()
            s2_word = raw["sense2_word"].strip().lower()
            c1_sent = raw["c1_sent"].strip()
            c2_sent = raw["c2_sent"].strip()

            if pos == "verb":
                s1_sent = f"To {target_word} means to {s1_word}."
                s2_sent = f"To {target_word} means to {s2_word}."
            elif pos in ("adjective", "adverb"):
                s1_sent = f"Being {target_word} means being {s1_word}."
                s2_sent = f"Being {target_word} means being {s2_word}."
            else:  # noun
                s1_sent = f"A {target_word} is a {s1_word}."
                s2_sent = f"A {target_word} is a {s2_word}."

            return {
                "word": target_word,
                "pos": pos,
                "c1_word": target_word,
                "c2_word": target_word,
                "s1_word": s1_word,
                "s2_word": s2_word,
                "c1_sent": c1_sent,
                "c2_sent": c2_sent,
                "def": {
                    "def1_sent": s1_sent, "def1_word": target_word,
                    "def2_sent": s2_sent, "def2_word": target_word,
                }
            }

        except ValueError as ve:
            raise ve
        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Generation failed after {max_retries} attempts: {e}")


# 2. FLAN-T5 Encoder

@st.cache_resource(show_spinner=f"Loading {MODEL_DISPLAY_NAME} encoder...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    full_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
    model = full_model.get_encoder()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def find_probe_token_index(tokenizer, sentence: str, probe_word: str):
    ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    norm_tokens = [t.replace("\u2581", "").lower() for t in raw_tokens]
    target = probe_word.lower().strip()

    exact_matches = [i for i, t in enumerate(norm_tokens) if t == target]
    if exact_matches:
        return exact_matches[-1]

    for start in range(len(norm_tokens)):
        acc = ""
        for end in range(start, len(norm_tokens)):
            acc += norm_tokens[end]
            if acc == target:
                return end
            if not target.startswith(acc):
                break
    raise ValueError(f"Probe word '{probe_word}' not found in tokens: {raw_tokens}")


@torch.no_grad()
def get_layerwise_probe_vectors(tokenizer, model, device, sentence: str, probe_word: str):
    num_layers = model.config.num_layers
    probe_idx = find_probe_token_index(tokenizer, sentence, probe_word)
    enc = tokenizer(sentence, return_tensors="pt").to(device)
    outputs = model(**enc, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    vectors = []
    # Index 0 = Input Embedding Layer (Layer 0), 1..num_layers = Encoder Layers
    for hs_idx in range(0, num_layers + 1):
        h = hidden_states[hs_idx]
        vectors.append(h[0, probe_idx, :].detach())
    return torch.stack(vectors)


def compute_centered_similarity(t1, t2):
    return F.cosine_similarity(t1.unsqueeze(1), t2.unsqueeze(0), dim=2).cpu().numpy()


# 3. Flowchart & Line Plot Rendering

def _plot_diamond(ax, matrix, word_label, cmap="viridis"):
    n = matrix.shape[0]
    i = np.arange(n + 1)
    j = np.arange(n + 1)
    I, J = np.meshgrid(i, j, indexing="ij")
    X = (J - I).astype(float)
    Y = -(I + J).astype(float)
    pcm = ax.pcolormesh(X, Y, matrix, cmap=cmap, shading="auto")
    ax.set_aspect("equal")
    ax.axis("off")

    ticks_to_show = np.linspace(0, n - 1, num=min(6, n), dtype=int).tolist()
    for idx in ticks_to_show:
        i_idx = idx + 0.5
        ax.text(-i_idx - 0.8, -i_idx, str(idx), ha="right", va="center", fontsize=8, color="#333333")
        ax.text(i_idx + 0.8, -i_idx, str(idx), ha="left", va="center", fontsize=8, color="#333333")

    ax.text(-n / 2 - 2, -n / 2, f'Layers 0-{n-1} ("{word_label}")', ha="right", va="center", fontsize=9, fontweight="bold", rotation=45)
    ax.text(n / 2 + 2, -n / 2, f'Layers 0-{n-1} ("{word_label}")', ha="left", va="center", fontsize=9, fontweight="bold", rotation=-45)

    return pcm


def render_flowchart(word_label, s1_label, s2_label, mat_self, mat_s1, mat_s2, probe_title_text) -> bytes:
    fig = plt.figure(figsize=(10, 10))
    
    gs = fig.add_gridspec(
        2, 3, 
        height_ratios=[1.15, 1], 
        width_ratios=[1, 1, 0.05], 
        hspace=0.45, 
        wspace=0.35
    )

    probe_vmin = min(mat_s1.min(), mat_s2.min())
    probe_vmax = max(mat_s1.max(), mat_s2.max())

    # Top Diamond Plot
    ax_top = fig.add_subplot(gs[0, :2])
    _plot_diamond(ax_top, mat_self, word_label)

    box_size = 5 
    total_states = mat_s1.shape[0] 
    ticks_to_show = np.linspace(0, total_states - 1, num=min(6, total_states), dtype=int).tolist()

    # Bottom Left Heatmap (Sense 1)
    ax_l = fig.add_subplot(gs[1, 0])
    mat_s1_flipped = np.fliplr(mat_s1)
    im_l = ax_l.imshow(mat_s1_flipped, cmap="viridis", origin="upper", vmin=probe_vmin, vmax=probe_vmax)
    
    ax_l.set_yticks(ticks_to_show)
    ax_l.set_yticklabels(ticks_to_show, fontsize=9)
    ax_l.set_ylabel(f'Probe "{word_label}" (Layer 0-{total_states-1})', fontsize=10, fontweight="bold")

    flipped_x_ticks_idx = [(total_states - 1) - t for t in ticks_to_show]
    ax_l.set_xticks(flipped_x_ticks_idx)
    ax_l.set_xticklabels(ticks_to_show, fontsize=9)
    ax_l.set_xlabel(f'"{word_label}" Sense 1 ({s1_label})', fontsize=10, fontweight="bold")

    rect_l = patches.Rectangle(
        (-0.5, total_states - box_size - 0.5), 
        box_size, box_size, 
        linewidth=2.5, edgecolor='red', facecolor='none'
    )
    ax_l.add_patch(rect_l)

    # Bottom Right Heatmap (Sense 2)
    ax_r = fig.add_subplot(gs[1, 1])
    ax_r.imshow(mat_s2, cmap="viridis", origin="upper", vmin=probe_vmin, vmax=probe_vmax)
    
    ax_r.set_yticks(ticks_to_show)
    ax_r.set_yticklabels(ticks_to_show, fontsize=9)
    ax_r.set_ylabel(f'Probe "{word_label}" (Layer 0-{total_states-1})', fontsize=10, fontweight="bold")

    ax_r.set_xticks(ticks_to_show)
    ax_r.set_xticklabels(ticks_to_show, fontsize=9)
    ax_r.set_xlabel(f'"{word_label}" Sense 2 ({s2_label})', fontsize=10, fontweight="bold")

    rect_r = patches.Rectangle(
        (total_states - box_size - 0.5, total_states - box_size - 0.5), 
        box_size, box_size, 
        linewidth=2.5, edgecolor='red', facecolor='none'
    )
    ax_r.add_patch(rect_r)

    # Colorbar
    cax = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(im_l, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Probe Cosine Similarity", fontsize=10, fontweight="bold")

    # Arrows
    arrow_props = dict(arrowstyle="-|>", color="#e88b8b", lw=4, mutation_scale=25, shrinkA=0, shrinkB=0)
    fig.patches.append(
        plt.matplotlib.patches.FancyArrowPatch(
            (0.35, 0.54), (0.22, 0.45), transform=fig.transFigure,
            connectionstyle="arc3,rad=-0.15", **arrow_props))
    fig.patches.append(
        plt.matplotlib.patches.FancyArrowPatch(
            (0.55, 0.54), (0.68, 0.45), transform=fig.transFigure,
            connectionstyle="arc3,rad=0.15", **arrow_props))

    # Super-Title
    full_title = f'{probe_title_text}\n\nSelf-Similarity Probing: "{word_label}" (Incl. Layer 0 Embeddings)'
    fig.suptitle(full_title, fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.92)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_line_plot(mat_s1, mat_s2, label1, label2, title) -> bytes:
    layers = np.arange(0, mat_s1.shape[0])
    diag_s1 = np.diag(mat_s1)
    diag_s2 = np.diag(mat_s2)

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.plot(layers, diag_s1, marker="o", linewidth=2, label=label1, color="#2b5c8f")
    ax.plot(layers, diag_s2, marker="s", linewidth=2, label=label2, color="#d95f02")

    # Highlight Layer 0
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.7)
    ax.text(0.2, ax.get_ylim()[0] + 0.1, "L0 (Input Emb)", fontsize=8, color="gray", rotation=90)

    ax.set_xlabel(f"{MODEL_DISPLAY_NAME} Encoder Layer (0 = Input Embeddings)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(-1, 1)
    
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# 4. Pipeline Execution

def process_data(item: dict):
    tokenizer, model, device = load_model()

    v_c1 = get_layerwise_probe_vectors(tokenizer, model, device, item["c1_sent"], item["c1_word"])
    v_c2 = get_layerwise_probe_vectors(tokenizer, model, device, item["c2_sent"], item["c2_word"])

    v_s1 = get_layerwise_probe_vectors(tokenizer, model, device, item["def"]["def1_sent"], item["def"]["def1_word"])
    v_s2 = get_layerwise_probe_vectors(tokenizer, model, device, item["def"]["def2_sent"], item["def"]["def2_word"])
    
    global_layer_mean = (v_c1 + v_c2 + v_s1 + v_s2) / 4.0
    c1 = v_c1 - global_layer_mean
    c2 = v_c2 - global_layer_mean
    s1 = v_s1 - global_layer_mean
    s2 = v_s2 - global_layer_mean

    mat_c1_self = compute_centered_similarity(c1, c1)
    mat_c1_s1 = compute_centered_similarity(c1, s1)
    mat_c1_s2 = compute_centered_similarity(c1, s2)

    mat_c2_self = compute_centered_similarity(c2, c2)
    mat_c2_s1 = compute_centered_similarity(c2, s1)
    mat_c2_s2 = compute_centered_similarity(c2, s2)

    w = item["word"]
    s1_w = item["s1_word"]
    s2_w = item["s2_word"]

    # Probe 1 Flowchart & Line Plot
    img1 = render_flowchart(
        w, s1_w, s2_w,
        mat_c1_self, mat_c1_s1, mat_c1_s2,
        f'Probe 1: "{item["c1_sent"]}"')
    
    lp1 = render_line_plot(
        mat_c1_s1, mat_c1_s2, 
        f'{w} in P1 <-> {w} in Sense 1 ({s1_w})', 
        f'{w} in P1 <-> {w} in Sense 2 ({s2_w})',
        f'Probe 1 Layer Progression (Layer 0 to {len(v_c1)-1}): "{item["c1_sent"]}"')

    # Probe 2 Flowchart & Line Plot
    img2 = render_flowchart(
        w, s1_w, s2_w,
        mat_c2_self, mat_c2_s1, mat_c2_s2,
        f'Probe 2: "{item["c2_sent"]}"')
    
    lp2 = render_line_plot(
        mat_c2_s1, mat_c2_s2, 
        f'{w} in P2 <-> {w} in Sense 1 ({s1_w})', 
        f'{w} in P2 <-> {w} in Sense 2 ({s2_w})',
        f'Probe 2 Layer Progression (Layer 0 to {len(v_c2)-1}): "{item["c2_sent"]}"')

    return img1, img2, lp1, lp2

# 5. Streamlit UI & State Management

st.title("Ambiguous Word Sense Visualizer")
st.caption(f"Layer-wise probing of {MODEL_DISPLAY_NAME} encoder representations (Layers 0 to N) across lexical ambiguity probing sentences.")

# Input controls
col_word, col_pos = st.columns([3, 1])
with col_word:
    word = st.text_input("Enter an ambiguous word", value="rare")
with col_pos:
    pos_option = st.selectbox(
        "Part of Speech", 
        options=["noun", "verb", "adjective", "adverb"], 
        index=2
    )

go = st.button("Generate Visualization", type="primary")

if go and word.strip():
    if "data_item" in st.session_state:
        del st.session_state["data_item"]
    if "viz_outputs" in st.session_state:
        del st.session_state["viz_outputs"]
        
    try:
        with st.spinner(f"Validating & generating coarse senses for '{word}' ({pos_option}) via OpenAI..."):
            item = generate_contexts(word.strip(), pos_option)
            st.session_state["data_item"] = item

            st.session_state["edit_c1_sent"] = item["c1_sent"]
            st.session_state["edit_c2_sent"] = item["c2_sent"]
            st.session_state["edit_s1_sent"] = item["def"]["def1_sent"]
            st.session_state["edit_s2_sent"] = item["def"]["def2_sent"]

        with st.spinner("Computing layer-wise embeddings (including Layer 0)..."):
            img1, img2, lp1, lp2 = process_data(item)
            st.session_state["viz_outputs"] = (img1, img2, lp1, lp2)

    except ValueError as ve:
        st.warning(f"[Invalid Combination / Validation Warning] {ve}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Display section if validated data is loaded
if "data_item" in st.session_state:
    item = st.session_state["data_item"]
    
    st.subheader(f'Word: "{item["word"]}" (POS: {item["pos"].upper()})')
    
    st.markdown("### Edit Sentences & Rerun")
    st.caption("Modify any sentence below to reactivate the regeneration button.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Probe 1**")
        edited_c1_sent = st.text_input("Probe 1 Sentence", key="edit_c1_sent")
        edited_s1_sent = st.text_input("Sense 1 Reference Sentence", key="edit_s1_sent")
    
    with c2:
        st.markdown("**Probe 2**")
        edited_c2_sent = st.text_input("Probe 2 Sentence", key="edit_c2_sent")
        edited_s2_sent = st.text_input("Sense 2 Reference Sentence", key="edit_s2_sent")

    has_changes = (
        edited_c1_sent != item['c1_sent'] or
        edited_c2_sent != item['c2_sent'] or
        edited_s1_sent != item['def']['def1_sent'] or
        edited_s2_sent != item['def']['def2_sent']
    )

    if has_changes:
        st.markdown("""
            <style>
            div.stButton > button[key="rerun_btn"] {
                background-color: #ff4b4b !important;
                color: white !important;
                border: none !important;
            }
            </style>
        """, unsafe_allow_html=True)

    rerun = st.button("Regenerate Visualization", disabled=not has_changes, key="rerun_btn")

    if rerun and has_changes:
        new_s1_word = extract_sense_word(edited_s1_sent, item["word"], item["s1_word"])
        new_s2_word = extract_sense_word(edited_s2_sent, item["word"], item["s2_word"])

        updated_item = {
            "word": item["word"],
            "pos": item["pos"],
            "c1_word": item["c1_word"],
            "c2_word": item["c2_word"],
            "s1_word": new_s1_word,
            "s2_word": new_s2_word,
            "c1_sent": edited_c1_sent,
            "c2_sent": edited_c2_sent,
            "def": {
                "def1_sent": edited_s1_sent, "def1_word": item["def"]["def1_word"],
                "def2_sent": edited_s2_sent, "def2_word": item["def"]["def2_word"],
            }
        }
        try:
            with st.spinner("Re-computing layer-wise embeddings with modified sentences..."):
                img1, img2, lp1, lp2 = process_data(updated_item)
                st.session_state["data_item"] = updated_item
                st.session_state["viz_outputs"] = (img1, img2, lp1, lp2)
                st.rerun()
        except Exception as e:
            st.error(f"Re-computation failed: {e}")

    if "viz_outputs" in st.session_state:
        img1, img2, lp1, lp2 = st.session_state["viz_outputs"]

        st.subheader("Flowcharts & Line Plots")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.image(img1, caption="Probe 1 Flowchart (Layers 0 to N)", width="stretch")
            st.download_button(
                label="Download Flowchart 1",
                data=img1,
                file_name=f"{item['word']}_probe1_flowchart.png",
                mime="image/png"
            )
            
            st.image(lp1, caption="Probe 1 Layer-wise Cosine Similarity (incl. L0)", width="stretch")
            st.download_button(
                label="Download Line Plot 1",
                data=lp1,
                file_name=f"{item['word']}_probe1_lineplot.png",
                mime="image/png"
            )

        with fc2:
            st.image(img2, caption="Probe 2 Flowchart (Layers 0 to N)", width="stretch")
            st.download_button(
                label="Download Flowchart 2",
                data=img2,
                file_name=f"{item['word']}_probe2_flowchart.png",
                mime="image/png"
            )
            
            st.image(lp2, caption="Probe 2 Layer-wise Cosine Similarity (incl. L0)", width="stretch")
            st.download_button(
                label="Download Line Plot 2",
                data=lp2,
                file_name=f"{item['word']}_probe2_lineplot.png",
                mime="image/png"
            )
