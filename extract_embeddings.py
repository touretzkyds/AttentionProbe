"""
extract_embeddings.py — Dual-Prompt Hidden-State Extraction & Visualisation
============================================================================

Refactor summary
----------------
* **Dual inference pipeline**: runs FLAN-T5-Large on Prompt 1 and Prompt 2
  independently, extracting ``encoder_hidden_states`` from each pass.

* **Side-by-side 32×32 matrix visualiser** (Mode 1):
    - ``plt.subplots(1, 2, figsize=(20, 8))`` — one row, two columns, matching
      the updated ``UI_CONFIG["figure_size"]`` in config.py.
    - Left subplot  → Prompt 1 hidden-state grid.
    - Right subplot → Prompt 2 hidden-state grid.

* **Token-length safety guards**: every hidden-state access is protected by
  ``min(token_idx, seq_len - 1)`` so mismatching sequence lengths between
  prompt variants never produce an ``IndexError``.

* **Live figure title**: ``fig.suptitle`` is updated on every redraw using the
  currently selected ``token_idx`` to show exactly which token from Prompt 1 is
  being compared against which token from Prompt 2, e.g.:
      "Comparing Representations | Prompt 1 Token: [says] vs Prompt 2 Token: [say]"

* All standard Matplotlib ``TextBox`` widgets are retained:
    - Layer selector for each subplot (above each matrix).
    - Min / Max colour-bar limit controls (below each matrix).
    - Shared token-index selector (bottom centre).

Usage
-----
    python extract_embeddings.py -mode 1

Modes
-----
    1   Side-by-side 32×32 hidden-state feature grids for two input prompts.
        (Modes 0 and 2 are reserved for future single-prompt extensions.)
"""

from __future__ import annotations

import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import TextBox
from transformers import T5ForConditionalGeneration, T5Tokenizer

from utils import is_float, is_int

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
# These are module-level so that the Matplotlib callback closures can mutate
# them without needing a class wrapper.

mode: Optional[int] = None

# Tokenised representations of the two input prompts.
tokens1: List[str] = []
tokens2: List[str] = []

# Encoder hidden states: list of tensors, one per layer.
# Shape of each tensor: (batch=1, seq_len, hidden_dim=1024).
encoder_hidden_states1 = None
encoder_hidden_states2 = None

# Sequence lengths (number of tokens including </s>).
seq_len1: int = 0
seq_len2: int = 0
num_layers: int = 0

# ---------------------------------------------------------------------------
# Matplotlib figure / axes references
# ---------------------------------------------------------------------------
fig: Optional[plt.Figure] = None
axs = None          # shape (2,) — axs[0] = Prompt 1, axs[1] = Prompt 2

im1 = None          # imshow handle for Prompt 1 matrix
im2 = None          # imshow handle for Prompt 2 matrix
cb1 = None          # colorbar handle for Prompt 1
cb2 = None          # colorbar handle for Prompt 2

# ---------------------------------------------------------------------------
# TextBox widgets — layer selectors
# ---------------------------------------------------------------------------
embedding1_layer_ax = None
embedding2_layer_ax = None
embedding1_layer: Optional[TextBox] = None
embedding2_layer: Optional[TextBox] = None

# ---------------------------------------------------------------------------
# TextBox widgets — min/max colour-bar controls
# ---------------------------------------------------------------------------
min_embedding1_label_ax = None
max_embedding1_label_ax = None
min_embedding2_label_ax = None
max_embedding2_label_ax = None

min_embedding1_label: Optional[TextBox] = None
max_embedding1_label: Optional[TextBox] = None
min_embedding2_label: Optional[TextBox] = None
max_embedding2_label: Optional[TextBox] = None

# ---------------------------------------------------------------------------
# TextBox widget — shared token-index selector
# ---------------------------------------------------------------------------
token_num_label_ax = None
token_num_label: Optional[TextBox] = None

# ---------------------------------------------------------------------------
# Active indices (mutated by callbacks)
# ---------------------------------------------------------------------------
emb1_layer_idx: int = 0
emb2_layer_idx: int = 0
token_idx: int = 0

# Colour-bar extents (updated on every redraw).
max_embedding1: float = float("-inf")
max_embedding2: float = float("-inf")
min_embedding1: float = float("inf")
min_embedding2: float = float("inf")


# ---------------------------------------------------------------------------
# TextBox callbacks
# ---------------------------------------------------------------------------

def _validate_layer(text: str) -> int:
    """Parse and validate a layer-index string; exits on bad input."""
    if not text.isdigit() or int(text) > 23:
        print(
            f"Error: '{text}' is not a valid layer index (expected 0–23). Exiting."
        )
        sys.exit(1)
    return int(text)


def submit_emb1_idx(text: str) -> None:
    """Called when the user submits a new layer index for Prompt 1."""
    global emb1_layer_idx
    emb1_layer_idx = _validate_layer(text)
    _redraw()


def submit_emb2_idx(text: str) -> None:
    """Called when the user submits a new layer index for Prompt 2."""
    global emb2_layer_idx
    emb2_layer_idx = _validate_layer(text)
    _redraw()


def submit_token_num_matrix(text: str) -> None:
    """Called when the user submits a new token index."""
    global token_idx
    max_bound = max(seq_len1, seq_len2) - 1
    if not text.isdigit() or int(text) > max_bound:
        print(
            f"Error: '{text}' is not a valid token index"
            f" (expected 0–{max_bound}). Exiting."
        )
        sys.exit(1)
    token_idx = int(text)
    _redraw()


def submit_emb1_min(text: str) -> None:
    """Update the lower colour-bar limit for Prompt 1."""
    global min_embedding1
    if not is_float(text):
        print(f"Error: '{text}' is not a valid float for Min. Exiting.")
        sys.exit(1)
    min_embedding1 = round(float(text), 2)
    if im1 is not None:
        im1.set_clim(min_embedding1, max_embedding1)
    plt.draw()


def submit_emb1_max(text: str) -> None:
    """Update the upper colour-bar limit for Prompt 1."""
    global max_embedding1
    if not is_float(text):
        print(f"Error: '{text}' is not a valid float for Max. Exiting.")
        sys.exit(1)
    max_embedding1 = round(float(text), 2)
    if im1 is not None:
        im1.set_clim(min_embedding1, max_embedding1)
    plt.draw()


def submit_emb2_min(text: str) -> None:
    """Update the lower colour-bar limit for Prompt 2."""
    global min_embedding2
    if not is_float(text):
        print(f"Error: '{text}' is not a valid float for Min. Exiting.")
        sys.exit(1)
    min_embedding2 = round(float(text), 2)
    if im2 is not None:
        im2.set_clim(min_embedding2, max_embedding2)
    plt.draw()


def submit_emb2_max(text: str) -> None:
    """Update the upper colour-bar limit for Prompt 2."""
    global max_embedding2
    if not is_float(text):
        print(f"Error: '{text}' is not a valid float for Max. Exiting.")
        sys.exit(1)
    max_embedding2 = round(float(text), 2)
    if im2 is not None:
        im2.set_clim(min_embedding2, max_embedding2)
    plt.draw()


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def _redraw() -> None:
    """Rebuild both matrix subplots from the current global state."""
    _render_matrix_subplots()


def _render_matrix_subplots() -> None:
    """
    Draw (or redraw) the side-by-side 32×32 hidden-state matrix visualisation.

    Layout
    ------
    * Row of TextBoxes above each subplot   → layer selector.
    * ``fig.suptitle``                       → live token comparison label.
    * Row of TextBoxes below each subplot   → Min / Max colour-bar limits.
    * Single TextBox at the bottom centre   → token-index selector.

    Safety
    ------
    Token access is guarded by ``min(token_idx, seq_len - 1)`` so that
    mismatching sequence lengths between the two prompts never raise an
    ``IndexError``.
    """
    global fig, axs
    global embedding1_layer_ax, embedding2_layer_ax
    global embedding1_layer, embedding2_layer
    global im1, im2, cb1, cb2
    global token_num_label_ax, token_num_label
    global min_embedding1_label_ax, max_embedding1_label_ax
    global min_embedding2_label_ax, max_embedding2_label_ax
    global min_embedding1_label, max_embedding1_label
    global min_embedding2_label, max_embedding2_label
    global max_embedding1, max_embedding2, min_embedding1, min_embedding2

    # ── First-time figure / widget construction ───────────────────────
    if fig is None:
        # 1 row × 2 cols, horizontal layout sized for standard monitors.
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        fig.subplots_adjust(
            left=0.07, right=0.93,
            top=0.82, bottom=0.18,
            wspace=0.35,
        )

        # Subplot titles
        axs[0].set_title("Prompt 1 — Hidden State Grid (32×32)", fontsize=12)
        axs[1].set_title("Prompt 2 — Hidden State Grid (32×32)", fontsize=12)

        # ── Layer-selector TextBoxes (above each subplot) ─────────────
        # Positions are in figure coordinates [left, bottom, width, height].
        embedding1_layer_ax = fig.add_axes([0.22, 0.88, 0.06, 0.05])
        embedding2_layer_ax = fig.add_axes([0.68, 0.88, 0.06, 0.05])

        embedding1_layer = TextBox(embedding1_layer_ax, label="Layer ", initial="0")
        embedding2_layer = TextBox(embedding2_layer_ax, label="Layer ", initial="0")

        for tb in (embedding1_layer, embedding2_layer):
            tb.label.set_fontsize(12)
            tb.text_disp.set_fontsize(12)

        embedding1_layer.on_submit(submit_emb1_idx)
        embedding2_layer.on_submit(submit_emb2_idx)

        # ── Token-index TextBox (bottom centre, shared) ───────────────
        token_num_label_ax = fig.add_axes([0.46, 0.04, 0.08, 0.05])
        token_num_label = TextBox(
            token_num_label_ax, label="Token Index ", initial="0"
        )
        token_num_label.label.set_fontsize(12)
        token_num_label.text_disp.set_fontsize(12)
        token_num_label.on_submit(submit_token_num_matrix)

        # ── Min / Max TextBoxes for Prompt 1 (bottom left) ───────────
        min_embedding1_label_ax = fig.add_axes([0.08, 0.04, 0.07, 0.05])
        max_embedding1_label_ax = fig.add_axes([0.18, 0.04, 0.07, 0.05])
        min_embedding1_label = TextBox(
            min_embedding1_label_ax, label="Min ", initial="0"
        )
        max_embedding1_label = TextBox(
            max_embedding1_label_ax, label="Max ", initial="0"
        )
        min_embedding1_label.on_submit(submit_emb1_min)
        max_embedding1_label.on_submit(submit_emb1_max)

        # ── Min / Max TextBoxes for Prompt 2 (bottom right) ──────────
        min_embedding2_label_ax = fig.add_axes([0.72, 0.04, 0.07, 0.05])
        max_embedding2_label_ax = fig.add_axes([0.82, 0.04, 0.07, 0.05])
        min_embedding2_label = TextBox(
            min_embedding2_label_ax, label="Min ", initial="0"
        )
        max_embedding2_label = TextBox(
            max_embedding2_label_ax, label="Max ", initial="0"
        )
        min_embedding2_label.on_submit(submit_emb2_min)
        max_embedding2_label.on_submit(submit_emb2_max)

    # ── Remove stale colorbars before redrawing ───────────────────────
    if cb1 is not None:
        cb1.remove()
        cb1 = None
    if cb2 is not None:
        cb2.remove()
        cb2 = None

    # ── Resolve which token label to display for each prompt ─────────
    # Safety guard: clamp token_idx to the valid range for each sequence
    # so that different-length prompts never cause an IndexError.
    safe_idx1 = min(token_idx, seq_len1 - 1)
    safe_idx2 = min(token_idx, seq_len2 - 1)

    t1_label = tokens1[safe_idx1] if seq_len1 > 0 else "[EMPTY]"
    t2_label = tokens2[safe_idx2] if seq_len2 > 0 else "[EMPTY]"

    # Live figure super-title — shows which tokens are being compared.
    fig.suptitle(
        f"Comparing Representations  |  "
        f"Prompt 1 Token: [{t1_label}]  vs  Prompt 2 Token: [{t2_label}]",
        fontsize=13,
        fontweight="bold",
    )

    # ── Extract and reshape Prompt 1 hidden-state vector ─────────────
    # encoder_hidden_states shape per layer: (1, seq_len, hidden_dim=1024).
    # We reshape the 1024-d vector into a 32×32 grid for display.
    vec1 = encoder_hidden_states1[emb1_layer_idx][0, safe_idx1, :]  # (1024,)
    grid1 = vec1.numpy().reshape(32, 32)

    axs[0].cla()
    axs[0].set_title(
        f"Prompt 1 — Hidden State Grid (32×32)\nLayer {emb1_layer_idx}",
        fontsize=11,
    )
    im1 = axs[0].imshow(grid1, cmap="viridis", aspect="auto")
    cb1 = fig.colorbar(im1, ax=axs[0], shrink=0.85, pad=0.04)

    # ── Extract and reshape Prompt 2 hidden-state vector ─────────────
    vec2 = encoder_hidden_states2[emb2_layer_idx][0, safe_idx2, :]  # (1024,)
    grid2 = vec2.numpy().reshape(32, 32)

    axs[1].cla()
    axs[1].set_title(
        f"Prompt 2 — Hidden State Grid (32×32)\nLayer {emb2_layer_idx}",
        fontsize=11,
    )
    im2 = axs[1].imshow(grid2, cmap="viridis", aspect="auto")
    cb2 = fig.colorbar(im2, ax=axs[1], shrink=0.85, pad=0.04)

    # ── Sync Min / Max TextBox labels to current clim values ─────────
    min_embedding1, max_embedding1 = im1.get_clim()
    min_embedding2, max_embedding2 = im2.get_clim()

    min_embedding1_label.set_val(str(round(min_embedding1, 2)))
    max_embedding1_label.set_val(str(round(max_embedding1, 2)))
    min_embedding2_label.set_val(str(round(min_embedding2, 2)))
    max_embedding2_label.set_val(str(round(max_embedding2, 2)))

    plt.draw()
    plt.show()


# ---------------------------------------------------------------------------
# Deprecated / stub modes
# ---------------------------------------------------------------------------

def cosine_sim_lineplot() -> None:
    """Mode 0 — single-prompt cosine-similarity line graph (deprecated)."""
    print(
        "Error: Mode 0 (single-prompt cosine-similarity) has been superseded by the"
        " dual-pipeline Mode 1 visualiser. Exiting."
    )
    sys.exit(1)


def matrix_cosine_sim_visualization() -> None:
    """Mode 2 — 24×24 cross-layer cosine-similarity matrix (reserved)."""
    print(
        "Error: Mode 2 is reserved for future cross-prompt similarity visualisation."
        " Exiting."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

def print_usage() -> None:
    """Print command-line usage information."""
    print("\nUsage: python extract_embeddings.py -mode <mode_number>")
    print("\nSupported modes:")
    print(
        "  1   Side-by-side 32×32 hidden-state feature grids.\n"
        "      Enter two prompts interactively to compare their encoder\n"
        "      representations layer-by-layer and token-by-token.\n"
        "      Example contrast pairs:\n"
        "        • 'The boy says everything' vs 'They say everything'\n"
        "        • 'He scratched them'       vs 'He scratched it'\n"
        "        • 'She loves them'          vs 'She loves it'\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, run dual inference, and launch the visualiser."""
    global mode
    global tokens1, tokens2
    global encoder_hidden_states1, encoder_hidden_states2
    global seq_len1, seq_len2, num_layers

    # ── Argument parsing ──────────────────────────────────────────────
    if len(sys.argv) < 3 or sys.argv[1] != "-mode":
        print_usage()
        sys.exit(1)

    if not is_int(sys.argv[2]):
        print_usage()
        sys.exit(1)

    mode = int(sys.argv[2])
    if mode != 1:
        print(
            "Error: Only Mode 1 is currently supported in this dual-pipeline build.\n"
            "       Run with '-mode 1'."
        )
        sys.exit(1)

    # ── Model loading ─────────────────────────────────────────────────
    print("\n─── Loading FLAN-T5-Large weights ──────────────────────────")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model.eval()
    print("Model loaded successfully.\n")

    # ── Prompt ingestion ──────────────────────────────────────────────
    print("[Dual-Prompt Hidden-State Comparative Extraction]")
    print(
        "Enter two prompts to compare (e.g. 'He says it' vs 'They say it').\n"
        "Type a sentence and press Enter for each prompt.\n"
    )
    user_input1 = input("Prompt 1: ").strip()
    user_input2 = input("Prompt 2: ").strip()

    if not user_input1 or not user_input2:
        print("Error: Both prompts must be non-empty. Exiting.")
        sys.exit(1)

    # ── Inference — Prompt 1 ──────────────────────────────────────────
    print(f"\nRunning inference on Prompt 1: '{user_input1}'…")
    inputs1 = tokenizer(user_input1, return_tensors="pt")
    with torch.no_grad():
        outputs1 = model.generate(
            input_ids=inputs1.input_ids,
            attention_mask=inputs1.attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_length=20,
        )
    tokens1 = tokenizer.convert_ids_to_tokens(inputs1.input_ids[0])
    encoder_hidden_states1 = outputs1.encoder_hidden_states
    # encoder_hidden_states1 is a tuple of (num_layers + 1) tensors.
    seq_len1 = encoder_hidden_states1[0].shape[1]

    # ── Inference — Prompt 2 ──────────────────────────────────────────
    print(f"Running inference on Prompt 2: '{user_input2}'…")
    inputs2 = tokenizer(user_input2, return_tensors="pt")
    with torch.no_grad():
        outputs2 = model.generate(
            input_ids=inputs2.input_ids,
            attention_mask=inputs2.attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_length=20,
        )
    tokens2 = tokenizer.convert_ids_to_tokens(inputs2.input_ids[0])
    encoder_hidden_states2 = outputs2.encoder_hidden_states
    seq_len2 = encoder_hidden_states2[0].shape[1]

    num_layers = len(encoder_hidden_states1)

    # ── Print token grids for reference ──────────────────────────────
    print(f"\nPrompt 1 token grid  ({seq_len1} tokens): {tokens1}")
    print(f"Prompt 2 token grid  ({seq_len2} tokens): {tokens2}")
    print(
        f"\nLayers available : 0 – {num_layers - 1}"
        f"\nToken indices    : 0 – {max(seq_len1, seq_len2) - 1}"
        f"  (mismatched lengths are handled automatically)\n"
    )

    # ── Launch the visualiser ─────────────────────────────────────────
    _render_matrix_subplots()


if __name__ == "__main__":
    main()