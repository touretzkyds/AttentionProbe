"""
Configuration settings for the AttentionProbe application.
"""

# Model Configuration
MODEL_NAME = "google/flan-t5-large"
DEVICE = "cpu"

# Demo Configuration
DEMO_CONFIGS = {
    "pronoun_resolution": {
        "name": "Demo 1",
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
        "auto_generate_prompt2": True,
        "need_prompt2": True,
        "default_prompt1": "The man showed the woman his jacket. Who owned the jacket, the man or the woman?",
        "default_prompt2": "The man showed the woman her jacket. Who owned the jacket, the man or the woman?",
        "validation_message": "This prompt must contain one of the following pronouns once: his, her",
        "interesting_heads": [(0, 15), (2, 6), (2, 8), (2, 9), (3, 6), (3, 9)]
    },
    "number_agreement": {
        "name": "Demo 2", 
        "description": "In this demo, we will focus on attention heads that pay attention to number agreement.",
        "description-details": "We will ask you to input one prompt of your own. Further guidelines are below: \n"
                               "1. Make sure to include in your prompt exactly one occurence of the pronouns: them, it. \n"
                               "2. Please keep this prompt relatively short and simple for best visualization. \n"
                               "3. Your prompt can include a question if you would like. \n"
                               "Example prompt: A man walked into a room with two cats and a refrigerator. He scratched them. What did the man scratch? \n"
                               "To use the example prompt as your prompt, type 0 and hit enter when you are asked to input your prompt. \n",
        "keywords": ["them", "it"],
        "max_keyword_count": 1,
        "auto_generate_prompt2": True,
        "need_prompt2": True,
        "default_prompt1": "A man walked into a room with two cats and a refrigerator. He scratched them. What did the man scratch?",
        "default_prompt2": "A man walked into a room with two cats and a refrigerator. He scratched it. What did the man scratch?",
        "validation_message": "This prompt must contain one of the following pronouns once: them, it",
        "interesting_heads": [(3, 9), (6, 14), (10, 9), (11, 15), (16, 10), (22, 14)]
    },
    "noun_phrases": {
        "name": "Demo 3",
        "description": "In this demo, we will focus on attention heads that identify noun phrases, mapping the phrase to the first 'the' in the noun phrase",
        "description-details": "We will ask you to input one prompt of your own. Further guidelines are below: \n"
                               "1. Make sure to include in your prompt at least one noun phrase that begins with 'the' \n"
                               "2. Please keep this prompt relatively short and simple for best visualization. \n"
                               "3. Your prompt can include a question if you would like. \n"
                               "Example prompt: The big white fluffy cat walked down the long black road in the morning.\n"
                               "To use the example prompt as your prompt, type 0 and hit enter when you are asked to input your prompt. \n",
        "keywords": ['the'],
        "max_keyword_count": -1, # this represents unlimited occurrences of "the" are allowed.
        "auto_generate_prompt2": False,
        "need_prompt2": False,
        "default_prompt1": "The big white fluffy cat walked down the long black road in the morning.",
        "default_prompt2": "The big white fluffy cat walked down the long black road in the morning.",
        "validation_message": "Your prompt must include at least one noun phrase starting with 'the'. An example of a noun phrase: The big white fluffy cat",
        "interesting_heads": [(7, 14), (8, 3), (8, 12), (8, 14), (9, 14), (10, 3), (10, 12), (11, 3)]
    },
    "prep_phrase_attach": {
        "name": "Demo 4",
        "description": "In this demo, we will focus on attention heads that perform prepositional phrase attachment.",
        "description-details": "We will ask you to input two prompts of your own. Further guidelines are below: \n"
                               "1. Make sure to include in each prompt exactly one occurrence of the following prepositions: in, with, for \n"
                               "2. Please keep your prompts relatively short and simple for best visualization. \n"
                               "3. Your prompts can include a question if you would like. \n"
                               "4. Your two prompts should be identical except for one word. This change should demonstrate a change in prepositional \n"
                               "   phrase attachment between the two sentences. \n"
                               "Example prompt 1: They discussed the plan for hours. What was the plan for?\n"
                               "Example prompt 2: They discussed the plan for dinner. What was the plan for?\n"
                               "To use the example prompt as your prompt, type 0 and hit enter when you are asked to input your prompt.\n ",
        "keywords": ["in", "with", "for"],
        "max_keyword_count": 1,
        "auto_generate_prompt2": False,
        "need_prompt2": True,
        "default_prompt1": "They discussed the plan for hours. What was the plan for?",
        "default_prompt2": "They discussed the plan for dinner. What was the plan for?",
        "validation_message": "This prompt must contain one of the following prepositions: in, with, for",
        "interesting_heads": [(12, 8), (14, 8), (16, 10), (18, 12), (21, 10)]
    }
}

# Embedding Visualization Configuration
EMBED_CONFIGS = {
    "introduction": "Welcome to the Embedding Visualization Demo! In this demo, we provide different modes of \n visualization"
                    " for examining how embeddings change through the layers of the FLAN-T5-large model.",
    "description-details-0": "You are currently in mode 0: plots cosine similarity between embedding at Layer 0 and the"
                             " embedding at every other layer via a line graph. In this mode, we will ask for an input below. Please"
                             " keep in mind that the visualization works best with shorter inputs.",
    "description-details-1": "You are currently in mode 1: represents embeddings as 32x32 matrices. This is an exploratory mode, "
                              "where you are given two 32x32 embedding matrices. Navigation instructions below: \n"
                             "1. You are able to navigate to embeddings at different layers using the text boxes above each matrix \n"
                             "2. You can change the min and max values of the colorbar display using the two text boxes below each matrix, labeled min and max \n"
                             "3. You can navigate to different tokens from your input sequence using the text box at the bottom of the screen. Note the tokenized \n"
                             "   sequence of your input will be printed after you give your input for your reference.",
    "description-details-2": "You are currently in mode 2: a 24x24 matrix that plots cosine similarity between every layer and every other layer. "
                             "Navigation instructions below: \n"
                             "1. You can navigate to different tokens from your input sequence using the text box at the bottom of the screen. Note the tokenized \n"
                             "   sequence of your input will be printed after you give your input for your reference."
}

# UI Configuration
UI_CONFIG = {
    "max_generation_length": 20,
    "slider_range": (0.0, 1.0),
    "slider_default": 1.0,
    "figure_size": (50, 8),
    "font_size": 10,
    "highlight_color": "red",
    "normal_color": "black"
}

COMMON_MESSAGES = {
    "navigation_msg": "To move to the next attention head, please press the right (->) arrow key on your keyboard.\n"
                           "To move to the previous attention head, please press the left (<-) arrow key on your keyboard."
}

# Validation Rules
VALIDATION_RULES = {
    "max_pronoun_count": 1,
    "required_pronouns": ["his", "her", "them", "it"]
} 