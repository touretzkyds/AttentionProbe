"""
Configuration settings for the AttentionProbe application.

Changelog (refactor):
  - number_agreement: broadened 'keywords' to cover subject/object pronouns and
    verb-agreement tokens so custom_pairs and verb_violation modes can pass freeform
    prompts without hitting the single-keyword hard-limit guard.
  - number_agreement: 'max_keyword_count' set to -1 (unlimited) to disable the
    regex counting check that crashed on multi-token custom inputs.
  - number_agreement: 'auto_generate_prompt2' set to False — the pipeline now
    expects the user to supply both prompts manually.
  - number_agreement: 'description-details' updated to guide the two-prompt flow.
  - UI_CONFIG: 'figure_size' changed to (20, 8) for a clean horizontal side-by-side
    32×32 matrix layout on standard monitors.
"""

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "google/flan-t5-large"
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Demo Configurations
# ---------------------------------------------------------------------------
DEMO_CONFIGS = {
    "pronoun_resolution": {
        "name": "Demo 1",
<<<<<<< Updated upstream
        "description": "In this demo, we will focus on attention heads that perform pronoun resolution.",
        "description-details": "We will ask you to input one prompt of your own. Further guidelines are below: \n"
                                "1. Make sure to include in your prompt exactly one occurence of the pronouns: his, her. \n"
                                "2. Please keep this prompt relatively short and simple for best visualization. \n"
                                "3. Your prompt can include a question if you would like. \n"
                                "Example prompt: The man showed the woman his jacket. Who owned the jacket, the man or the woman? \n"
                                "To use the example prompt as your prompt, type 0 and hit enter when you are asked to input your prompt. "
                               "\n\n",
        "keywords": ["him", "her", "his", "hers"],
        "PRONOUN_MAP" : {
            "him": "her",
            "her": "his",
            "his": "her",
            "hers": "his"  
        },   
        "max_keyword_count": 5,
=======
        "description": (
            "In this demo, we will focus on attention heads that perform pronoun resolution."
        ),
        "description-details": (
            "We will ask you to input one prompt of your own. Further guidelines are below:\n"
            "1. Make sure to include in your prompt exactly one occurrence of the pronouns:"
            " his, her.\n"
            "2. Please keep this prompt relatively short and simple for best visualization.\n"
            "3. Your prompt can include a question if you would like.\n"
            "Example prompt: The man showed the woman his jacket."
            " Who owned the jacket, the man or the woman?\n"
            "To use the example prompt as your prompt, type 0 and hit enter when you are"
            " asked to input your prompt.\n\n"
        ),
        "keywords": ["his", "her"],
        "max_keyword_count": 1,
>>>>>>> Stashed changes
        "auto_generate_prompt2": True,
        "need_prompt2": True,
        "default_prompt1": (
            "The man showed the woman his jacket."
            " Who owned the jacket, the man or the woman?"
        ),
        "default_prompt2": (
            "The man showed the woman her jacket."
            " Who owned the jacket, the man or the woman?"
        ),
        "validation_message": (
            "This prompt must contain one of the following pronouns once: his, her"
        ),
        "interesting_heads": [
            (0, 15), (2, 6), (2, 8), (2, 9), (3, 6), (3, 9)
        ],
    },

    # ------------------------------------------------------------------
    # number_agreement  ← REFACTORED
    # ------------------------------------------------------------------
    "number_agreement": {
        "name": "Demo 2",
        "description": (
            "In this demo, we will focus on attention heads that handle number agreement"
            " and syntactic verb-form selection."
        ),
        # Updated description instructs users to supply TWO custom prompts manually.
        "description-details": (
            "We will ask you to input TWO custom prompts for comparison. Guidelines:\n"
            "1. Prompt 1 and Prompt 2 should form a minimal contrast pair — ideally identical"
            " except for one pronoun or verb token.\n"
            "   Examples of valid contrast pairs:\n"
            "     • Object-pronoun   : 'He scratched them.' vs 'He scratched it.'\n"
            "     • Subject-pronoun  : 'He says everything.' vs 'They say everything.'\n"
            "     • Verb violation   : 'The boy says it.' vs 'The boy say it.'\n"
            "2. Please keep both prompts relatively short for clearest visualisation.\n"
            "3. Prompts may include a follow-up question.\n"
            "4. Type 0 at either prompt to load the built-in default for that slot.\n\n"
        ),
        # Broadened token set covers object pronouns, subject pronouns, and verb forms
        # so the loose validation guard in custom_pairs / verb_violation mode won't
        # reject valid freeform inputs.
        "keywords": ["them", "it", "he", "they", "she", "say", "says"],
        # -1 disables the keyword-count hard limit entirely; freeform sentences are allowed.
        "max_keyword_count": -1,
        # False → pipeline expects the user to type both prompts; no auto-swap logic runs.
        "auto_generate_prompt2": False,
        "need_prompt2": True,
        "default_prompt1": (
            "A man walked into a room with two cats and a refrigerator."
            " He scratched them. What did the man scratch?"
        ),
        "default_prompt2": (
            "A man walked into a room with two cats and a refrigerator."
            " He scratched it. What did the man scratch?"
        ),
        "validation_message": (
            "Please enter a non-empty sentence (minimum 5 characters)."
        ),
        "interesting_heads": [
            (3, 9), (6, 14), (10, 9), (11, 15), (16, 10), (22, 14)
        ],
    },

    "noun_phrases": {
        "name": "Demo 3",
        "description": (
            "In this demo, we will focus on attention heads that identify noun phrases,"
            " mapping the phrase to the first 'the' in the noun phrase."
        ),
        "description-details": (
            "We will ask you to input one prompt of your own. Further guidelines are below:\n"
            "1. Make sure to include in your prompt at least one noun phrase that begins"
            " with 'the'.\n"
            "2. Please keep this prompt relatively short and simple for best visualization.\n"
            "3. Your prompt can include a question if you would like.\n"
            "Example prompt: The big white fluffy cat walked down the long black road"
            " in the morning.\n"
            "To use the example prompt as your prompt, type 0 and hit enter when you are"
            " asked to input your prompt.\n"
        ),
        "keywords": ["the"],
        # -1 represents unlimited occurrences of 'the' are allowed.
        "max_keyword_count": -1,
        "auto_generate_prompt2": False,
        "need_prompt2": False,
        "default_prompt1": (
            "The big white fluffy cat walked down the long black road in the morning."
        ),
        "default_prompt2": (
            "The big white fluffy cat walked down the long black road in the morning."
        ),
        "validation_message": (
            "Your prompt must include at least one noun phrase starting with 'the'."
            " An example of a noun phrase: The big white fluffy cat"
        ),
        "interesting_heads": [
            (7, 14), (8, 3), (8, 12), (8, 14), (9, 14), (10, 3), (10, 12), (11, 3)
        ],
    },

    "prep_phrase_attach": {
        "name": "Demo 4",
        "description": (
            "In this demo, we will focus on attention heads that perform"
            " prepositional phrase attachment."
        ),
        "description-details": (
            "We will ask you to input two prompts of your own. Further guidelines are below:\n"
            "1. Make sure to include in each prompt exactly one occurrence of the following"
            " prepositions: in, with, for.\n"
            "2. Please keep your prompts relatively short and simple for best visualization.\n"
            "3. Your prompts can include a question if you would like.\n"
            "4. Your two prompts should be identical except for one word. This change should"
            " demonstrate a change in prepositional phrase attachment between the two"
            " sentences.\n"
            "Example prompt 1: They discussed the plan for hours. What was the plan for?\n"
            "Example prompt 2: They discussed the plan for dinner. What was the plan for?\n"
            "To use the example prompt as your prompt, type 0 and hit enter when you are"
            " asked to input your prompt.\n"
        ),
        "keywords": ["in", "with", "for"],
        "max_keyword_count": 1,
        "auto_generate_prompt2": False,
        "need_prompt2": True,
        "default_prompt1": "They discussed the plan for hours. What was the plan for?",
        "default_prompt2": "They discussed the plan for dinner. What was the plan for?",
        "validation_message": (
            "This prompt must contain one of the following prepositions: in, with, for"
        ),
        "interesting_heads": [
            (12, 8), (14, 8), (16, 10), (18, 12), (21, 10)
        ],
    },
}

# ---------------------------------------------------------------------------
# Embedding Visualisation Configuration
# ---------------------------------------------------------------------------
EMBED_CONFIGS = {
    "introduction": (
        "Welcome to the Embedding Visualisation Demo! In this demo, we provide different"
        " modes of visualisation for examining how embeddings change through the layers of"
        " the FLAN-T5-large model."
    ),
    "description-details-0": (
        "You are currently in mode 0: plots cosine similarity between the embedding at"
        " Layer 0 and the embedding at every other layer via a line graph. In this mode,"
        " we will ask for an input below. Please keep in mind that the visualisation works"
        " best with shorter inputs."
    ),
    "description-details-1": (
        "You are currently in mode 1: represents embeddings as 32×32 matrices."
        " This is an exploratory mode where you are given two 32×32 embedding matrices.\n"
        "Navigation instructions:\n"
        "1. Navigate to embeddings at different layers using the text boxes above each"
        " matrix.\n"
        "2. Change the min/max values of the colour-bar display using the two text boxes"
        " below each matrix, labelled Min and Max.\n"
        "3. Navigate to different tokens from your input sequence using the text box at"
        " the bottom of the screen. Note: the tokenised sequence of your input will be"
        " printed after you give your input for your reference."
    ),
    "description-details-2": (
        "You are currently in mode 2: a 24×24 matrix that plots cosine similarity between"
        " every layer and every other layer.\n"
        "Navigation instructions:\n"
        "1. Navigate to different tokens from your input sequence using the text box at"
        " the bottom of the screen. Note: the tokenised sequence of your input will be"
        " printed after you give your input for your reference."
    ),
}

# ---------------------------------------------------------------------------
# UI Configuration
# CHANGED: figure_size → (20, 8) for a clean horizontal side-by-side layout.
# ---------------------------------------------------------------------------
UI_CONFIG = {
    "max_generation_length": 20,
    "slider_range": (0.0, 1.0),
    "slider_default": 1.0,
    # (20, 8) gives a wide horizontal canvas so two 32×32 matrices render
    # comfortably on a standard 1080p / 1440p monitor without clipping.
    "figure_size": (20, 8),
    "font_size": 10,
    "highlight_color": "red",
    "normal_color": "black",
}

COMMON_MESSAGES = {
    "navigation_msg": (
        "To move to the next attention head, please press the right (→) arrow key.\n"
        "To move to the previous attention head, please press the left (←) arrow key."
    ),
}

# ---------------------------------------------------------------------------
# Validation Rules
# ---------------------------------------------------------------------------
VALIDATION_RULES = {
    "max_pronoun_count": 1,
    "required_pronouns": ["his", "her", "them", "it"],
}