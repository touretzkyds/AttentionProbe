"""
Ambiguous Word Sense Visualizer
================================
Layer-wise probing of FLAN-T5-Large encoder representations across lexical ambiguity contexts.
"""

import os
import json
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from transformers import T5EncoderModel, AutoTokenizer
from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIG & ENVIRONMENT API KEY HANDLING
# ---------------------------------------------------------------------------
# Read key safely using os.getenv without hardcoding secrets in git repo
ENV_API_KEY = os.getenv("OPENAI_API_KEY", "")

OPENAI_MODEL = "gpt-4o-mini"
T5_MODEL_NAME = "google/flan-t5-large"

st.set_page_config(page_title="Ambiguous Word Sense Visualizer", layout="wide")

# Sidebar Fallback UI for User/Graders
st.sidebar.title("Configuration")
user_api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    value=ENV_API_KEY, 
    type="password",
    help="Reads from os.getenv('OPENAI_API_KEY') by default. You can also paste it manually here."
)

# Active key preference: Sidebar input > Environment variable
ACTIVE_API_KEY = user_api_key if user_api_key else ENV_API_KEY


# ---------------------------------------------------------------------------
# 1. JSON Schema & System Prompt (Enforcing POS Consistency)
# ---------------------------------------------------------------------------
AMBIGUITY_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ambiguous_word_senses",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "word": {"type": "string"},
                "pos": {
                    "type": "string",
                    "enum": ["noun", "verb", "adjective"],
                    "description": "Part of speech shared by BOTH senses"
                },
                "sense1_word": {
                    "type": "string",
                    "pattern": "^[a-zA-Z]+$",
                    "description": "Single-word synonym for sense 1"
                },
                "sense2_word": {
                    "type": "string",
                    "pattern": "^[a-zA-Z]+$",
                    "description": "Single-word synonym for sense 2"
                },
                "c1_sent": {
                    "type": "string",
                    "description": "Short sentence (5-8 words) using the target word in sense 1."
                },
                "c2_sent": {
                    "type": "string",
                    "description": "Minimal pair sentence (5-8 words) using the target word in sense 2."
                }
            },
            "required": ["word", "pos", "sense1_word", "sense2_word", "c1_sent", "c2_sent"],
            "additionalProperties": False
        }
    }
}

SYSTEM_PROMPT = """You are a precise linguistics data generator for a lexical-ambiguity probing experiment.

Pick two distinct senses of the given word and produce:
1. sense1_word: A single-word common English synonym/hypernym for sense 1 (e.g. "device").
2. sense2_word: A single-word common English synonym/hypernym for sense 2 (e.g. "rodent").
3. c1_sent: A short (5-8 words) simple sentence using the target word in sense 1.
4. c2_sent: A short (5-8 words) simple sentence using the target word in sense 2.

CRITICAL POS RULE:
- BOTH senses MUST share the EXACT SAME Part of Speech (POS).
- If the word is tested as a NOUN, BOTH senses must be nouns (e.g., mouse -> rodent / device).
- If tested as a VERB, BOTH senses must be verbs (e.g., draw -> sketch / attract).
- NEVER mix a noun sense with a verb sense.

CRITICAL FORMAT RULES:
- BOTH sense1_word and sense2_word MUST be strictly single words (alphabetic characters only).
- Target word must appear verbatim in both c1_sent and c2_sent.
"""


def generate_contexts(word: str, max_retries: int = 3) -> dict:
    if not ACTIVE_API_KEY:
        st.error("❌ OpenAI API Key not found!")
        st.info("Please set the `OPENAI_API_KEY` environment variable in your terminal OR enter your key in the sidebar.")
        st.stop()
        
    client = OpenAI(api_key=ACTIVE_API_KEY)
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                response_format=AMBIGUITY_RESPONSE_SCHEMA,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Ambiguous word: {word}"},
                ],
            )
            raw = json.loads(resp.choices[0].message.content)
            
            target_word = raw["word"].strip()
            s1_word = raw["sense1_word"].strip().lower()
            s2_word = raw["sense2_word"].strip().lower()
            c1_sent = raw["c1_sent"].strip()
            c2_sent = raw["c2_sent"].strip()
            pos = raw["pos"].strip().lower()

            # Definitional anchor sentences
            if pos == "verb":
                def1_sent = f"To {target_word} means to {s1_word}."
                def2_sent = f"To {target_word} means to {s2_word}."
            else: # noun / default
                def1_sent = f"A {target_word} is a {s1_word}."
                def2_sent = f"A {target_word} is a {s2_word}."

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
                    "def1_sent": def1_sent, "def1_word": target_word,
                    "def2_sent": def2_sent, "def2_word": target_word,
                }
            }

        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Generation failed after {max_retries} attempts: {e}")


# ---------------------------------------------------------------------------
# 2. FLAN-T5 Encoder
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading FLAN-T5-Large encoder...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    model = T5EncoderModel.from_pretrained(T5_MODEL_NAME)
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
    for hs_idx in range(1, num_layers + 1):
        h = hidden_states[hs_idx]
        vectors.append(h[0, probe_idx, :].detach())
    return torch.stack(vectors)


def compute_centered_similarity(t1, t2):
    return F.cosine_similarity(t1.unsqueeze(1), t2.unsqueeze(0), dim=2).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Flowchart & Line Plot Rendering
# ---------------------------------------------------------------------------
def _plot_diamond(ax, matrix, cmap="viridis"):
    n = matrix.shape[0]
    i = np.arange(n + 1)
    j = np.arange(n + 1)
    I, J = np.meshgrid(i, j, indexing="ij")
    X = (J - I).astype(float)
    Y = -(I + J).astype(float)
    pcm = ax.pcolormesh(X, Y, matrix, cmap=cmap, shading="auto")
    ax.set_aspect("equal")
    ax.axis("off")

    # Scale Ticks 1..24 on Diamond edges
    ticks_to_show = [1, 6, 12, 18, 24]
    for idx in ticks_to_show:
        i_idx = idx - 0.5
        # Left edge (Layer 1 to 24 along Left axis)
        x_left = -i_idx
        y_left = -i_idx
        ax.text(x_left - 0.8, y_left, str(idx), ha="right", va="center", fontsize=8, color="#333333")

        # Right edge (Layer 1 to 24 along Right axis)
        x_right = i_idx
        y_right = -i_idx
        ax.text(x_right + 0.8, y_right, str(idx), ha="left", va="center", fontsize=8, color="#333333")

    ax.text(-n / 2 - 2, -n / 2, "Layer 1..24", ha="right", va="center", fontsize=9, fontweight="bold", rotation=45)
    ax.text(n / 2 + 2, -n / 2, "Layer 1..24", ha="left", va="center", fontsize=9, fontweight="bold", rotation=-45)

    return pcm


def render_flowchart(word_label, m1_label, m2_label, mat_self, mat_m1, mat_m2, title) -> bytes:
    fig = plt.figure(figsize=(10, 10))
    
    gs = fig.add_gridspec(
        2, 3, 
        height_ratios=[1.15, 1], 
        width_ratios=[1, 1, 0.05], 
        hspace=0.50, 
        wspace=0.35
    )

    probe_vmin = min(mat_m1.min(), mat_m2.min())
    probe_vmax = max(mat_m1.max(), mat_m2.max())

    # Top Diamond (OG Self-Similarity with Layer 1..24 ticks)
    ax_top = fig.add_subplot(gs[0, :2])
    _plot_diamond(ax_top, mat_self)
    ax_top.text(0, 6, f'Self-Similarity: "{word_label}"', ha="center", va="bottom", fontsize=16, fontweight="bold")

    box_size = 5  # Highlight area size (last 5 layers)
    num_layers = mat_m1.shape[0]  # 24 layers
    ticks_to_show = [1, 6, 12, 18, 24]  # Layer ticks

    # -------------------------------------------------------------------
    # Bottom Left Heatmap (Horizontal Flip)
    # -------------------------------------------------------------------
    ax_l = fig.add_subplot(gs[1, 0])
    mat_m1_flipped = np.fliplr(mat_m1)
    im_l = ax_l.imshow(mat_m1_flipped, cmap="viridis", origin="upper", vmin=probe_vmin, vmax=probe_vmax)
    
    # Y-axis Label & Ticks
    ax_l.set_yticks([t - 1 for t in ticks_to_show])
    ax_l.set_yticklabels(ticks_to_show, fontsize=9)
    ax_l.set_ylabel(f'Probe "{word_label}"', fontsize=11, fontweight="bold")

    # X-axis Label & Ticks (Flipped 24..1)
    flipped_x_ticks_idx = [num_layers - t for t in ticks_to_show]
    ax_l.set_xticks(flipped_x_ticks_idx)
    ax_l.set_xticklabels(ticks_to_show, fontsize=9)
    ax_l.set_xlabel(f'"{word_label}" {m1_label} sense', fontsize=11, fontweight="bold")

    # Red Box on Left Heatmap (Bottom-Left Corner)
    rect_l = patches.Rectangle(
        (-0.5, num_layers - box_size - 0.5), 
        box_size, box_size, 
        linewidth=3, edgecolor='red', facecolor='none'
    )
    ax_l.add_patch(rect_l)

    # -------------------------------------------------------------------
    # Bottom Right Heatmap (Standard Alignment)
    # -------------------------------------------------------------------
    ax_r = fig.add_subplot(gs[1, 1])
    ax_r.imshow(mat_m2, cmap="viridis", origin="upper", vmin=probe_vmin, vmax=probe_vmax)
    
    # Y-axis Label & Ticks
    ax_r.set_yticks([t - 1 for t in ticks_to_show])
    ax_r.set_yticklabels(ticks_to_show, fontsize=9)
    ax_r.set_ylabel(f'Probe "{word_label}"', fontsize=11, fontweight="bold")

    # X-axis Label & Ticks (Standard 1..24)
    ax_r.set_xticks([t - 1 for t in ticks_to_show])
    ax_r.set_xticklabels(ticks_to_show, fontsize=9)
    ax_r.set_xlabel(f'"{word_label}" {m2_label} sense', fontsize=11, fontweight="bold")

    # Red Box on Right Heatmap (Bottom-Right Corner)
    rect_r = patches.Rectangle(
        (num_layers - box_size - 0.5, num_layers - box_size - 0.5), 
        box_size, box_size, 
        linewidth=3, edgecolor='red', facecolor='none'
    )
    ax_r.add_patch(rect_r)

    # Colorbar
    cax = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(im_l, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Probe Cosine Similarity", fontsize=10, fontweight="bold")

    # Arrows
    arrow_props = dict(arrowstyle="-|>", color="#e88b8b", lw=5, mutation_scale=30, shrinkA=0, shrinkB=0)
    fig.patches.append(
        plt.matplotlib.patches.FancyArrowPatch(
            (0.35, 0.55), (0.22, 0.46), transform=fig.transFigure,
            connectionstyle="arc3,rad=-0.15", **arrow_props))
    fig.patches.append(
        plt.matplotlib.patches.FancyArrowPatch(
            (0.55, 0.55), (0.68, 0.46), transform=fig.transFigure,
            connectionstyle="arc3,rad=0.15", **arrow_props))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    
    # Clean padding adjustment (replaces tight_layout to prevent UserWarnings)
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.92)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_line_plot(mat_m1, mat_m2, label1, label2, title) -> bytes:
    layers = np.arange(1, mat_m1.shape[0] + 1)
    diag_m1 = np.diag(mat_m1)
    diag_m2 = np.diag(mat_m2)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(layers, diag_m1, marker="o", linewidth=2, label=label1, color="#2b5c8f")
    ax.plot(layers, diag_m2, marker="s", linewidth=2, label=label2, color="#d95f02")

    ax.set_xlabel("FLAN-T5 Encoder Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(layers)
    
    # Fixed Y-axis scale: -1 to 1
    ax.set_ylim(-1, 1)
    
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 4. Pipeline Execution
# ---------------------------------------------------------------------------
def process_word(word: str):
    tokenizer, model, device = load_model()

    with st.spinner(f"Generating senses for '{word}' via OpenAI..."):
        item = generate_contexts(word)

    with st.spinner("Computing layer-wise embeddings..."):
        # Base context vectors
        v_c1 = get_layerwise_probe_vectors(tokenizer, model, device, item["c1_sent"], item["c1_word"])
        v_c2 = get_layerwise_probe_vectors(tokenizer, model, device, item["c2_sent"], item["c2_word"])

        # Probe target_word inside definition anchor sentences
        v_m1 = get_layerwise_probe_vectors(tokenizer, model, device, item["def"]["def1_sent"], item["def"]["def1_word"])
        v_m2 = get_layerwise_probe_vectors(tokenizer, model, device, item["def"]["def2_sent"], item["def"]["def2_word"])
        
        # Mean centering across all contexts
        global_layer_mean = (v_c1 + v_c2 + v_m1 + v_m2) / 4.0
        c1 = v_c1 - global_layer_mean
        c2 = v_c2 - global_layer_mean
        m1 = v_m1 - global_layer_mean
        m2 = v_m2 - global_layer_mean

        mat_c1_self = compute_centered_similarity(c1, c1)
        mat_c1_m1 = compute_centered_similarity(c1, m1)
        mat_c1_m2 = compute_centered_similarity(c1, m2)

        mat_c2_self = compute_centered_similarity(c2, c2)
        mat_c2_m1 = compute_centered_similarity(c2, m1)
        mat_c2_m2 = compute_centered_similarity(c2, m2)

    w = item["word"]
    s1 = item["s1_word"]
    s2 = item["s2_word"]

    # Context 1 Flowchart & Line Plot
    img1 = render_flowchart(
        w, s1, s2,
        mat_c1_self, mat_c1_m1, mat_c1_m2,
        f'Context 1: "{item["c1_sent"]}"')
    
    lp1 = render_line_plot(
        mat_c1_m1, mat_c1_m2, 
        f'{w} in C1 <-> {w} in Def 1 ({s1})', 
        f'{w} in C1 <-> {w} in Def 2 ({s2})',
        f'Context 1 Layer Progression: "{item["c1_sent"]}"')

    # Context 2 Flowchart & Line Plot
    img2 = render_flowchart(
        w, s1, s2,
        mat_c2_self, mat_c2_m1, mat_c2_m2,
        f'Context 2: "{item["c2_sent"]}"')
    
    lp2 = render_line_plot(
        mat_c2_m1, mat_c2_m2, 
        f'{w} in C2 <-> {w} in Def 1 ({s1})', 
        f'{w} in C2 <-> {w} in Def 2 ({s2})',
        f'Context 2 Layer Progression: "{item["c2_sent"]}"')

    return item, img1, img2, lp1, lp2


# ---------------------------------------------------------------------------
# 5. Streamlit UI
# ---------------------------------------------------------------------------
st.title("Ambiguous Word Sense Visualizer")
st.caption("Layer-wise probing of FLAN-T5-Large encoder representations across lexical ambiguity contexts.")

word = st.text_input("Enter an ambiguous word", value="mouse")
go = st.button("Generate Visualization", type="primary")

if go and word.strip():
    try:
        item, img1, img2, lp1, lp2 = process_word(word.strip())

        st.subheader(f'Word: "{item["word"]}" (POS: {item["pos"].upper()})')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Context 1:** {item['c1_sent']}")
            st.markdown(f"- **Def 1 Anchor:** {item['def']['def1_sent']}")
            st.markdown(f"- **Def 2 Anchor:** {item['def']['def2_sent']}")
        with c2:
            st.markdown(f"**Context 2:** {item['c2_sent']}")
            st.markdown(f"- **Def 1 Anchor:** {item['def']['def1_sent']}")
            st.markdown(f"- **Def 2 Anchor:** {item['def']['def2_sent']}")

        st.subheader("Flowcharts & Line Plots")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.image(img1, caption="Context 1 Flowchart", width="stretch")
            st.download_button(
                label="📥 Download Flowchart 1",
                data=img1,
                file_name=f"{item['word']}_context1_flowchart.png",
                mime="image/png"
            )
            
            st.image(lp1, caption="Context 1 Layer-wise Cosine Similarity", width="stretch")
            st.download_button(
                label="📥 Download Line Plot 1",
                data=lp1,
                file_name=f"{item['word']}_context1_lineplot.png",
                mime="image/png"
            )

        with fc2:
            st.image(img2, caption="Context 2 Flowchart", width="stretch")
            st.download_button(
                label="📥 Download Flowchart 2",
                data=img2,
                file_name=f"{item['word']}_context2_flowchart.png",
                mime="image/png"
            )
            
            st.image(lp2, caption="Context 2 Layer-wise Cosine Similarity", width="stretch")
            st.download_button(
                label="📥 Download Line Plot 2",
                data=lp2,
                file_name=f"{item['word']}_context2_lineplot.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
