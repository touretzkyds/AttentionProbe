<<<<<<< Updated upstream
import subprocess
from typing import List, Tuple
from config import DEMO_CONFIGS, UI_CONFIG, VALIDATION_RULES, COMMON_MESSAGES
from utils import ModelManager, validate_sentence, generate_contrast_prompt
from attention_visualizer import AttentionVisualizer

class BaseDemo:
    """Base class for attention probe demonstrations."""
    
    def __init__(self, demo_type: str):
        """
        Initialize the demo with configuration.

        Args:
            demo_type: Type of demo ('pronoun_resolution' or 'number_agreement')
        """
        if demo_type not in DEMO_CONFIGS:
            raise ValueError(f"Unknown demo type: {demo_type}")
        self.demo_type = demo_type
        self.config = DEMO_CONFIGS[demo_type]
        self.common_messages = COMMON_MESSAGES
        self.model_manager = ModelManager("google/flan-t5-large")
        self.prompt1 = ""
        self.prompt2 = ""

    def execute_introduction(self):
        """Display the demo introduction."""
        print(f"Hello! Welcome to {self.config['name']}. In this demonstration, we will ask you to input two of your own prompts.")
        print("We will run your sentences on Google's FLAN-T5 Large model, and will show you interesting attention heads. \n\n")
        print(self.config['description'])
        print(self.config['description-details'])
        print(self.common_messages['navigation_msg'])

    def transition_description(self):
        """Display the transition message."""
        print("Now, we will present some of the notable attention heads. Press q to exit from the demonstration.")

    def get_user_prompt(self, prompt_number: int) -> str:
        """
        Get a valid prompt from the user.

        Args:
            prompt_number: Which prompt number this is (1 or 2)
        Returns:
            Valid user prompt
        """
        print(f"Please input your prompt below.")
        while True:
            prompt = input(f"What do you want for prompt {prompt_number}? ")
            # Allow using default prompt
            if prompt == "0":
                return self.config[f'default_prompt{prompt_number}']

            # Validate the prompt
            if validate_sentence(prompt, self.config['keywords'], self.config['max_keyword_count']):
                return prompt
            else:
                print("Your prompt does not satisfy the requirements. Please reenter a valid prompt below.")

    def generate_contrasting_prompt(self, original_prompt: str) -> str:
        """
        Generate a contrasting prompt from the original.

        Args:
            original_prompt: The original user prompt

        Returns:
            Contrasting prompt with pronoun swapped
        """
        try:
            prompt = generate_contrast_prompt(
                original_prompt,
                self.config['keywords'],
                self.config['auto_generate_prompt2'],
                self.config['PRONOUN_MAP']
            )
            if prompt == "":
                return self.get_user_prompt(2)
            return prompt
        except ValueError:
            # Fallback to default if generation fails
            return self.config['default_prompt2']

    def run_model_inference(self):
        """Run both prompts through the model and display results."""
        print("Let's run both your prompts through the model. Here is the output of the model below: ")
        for i, prompt in enumerate([self.prompt1, self.prompt2], 1):
            response = self.model_manager.generate_response(
                prompt,
                max_length=UI_CONFIG['max_generation_length']
            )
            print(f'{prompt} -> {response}')

    def launch_visualization(self):
        """Launch the attention visualization."""
        # Use the new visualizer instead of subprocess
        self.visualizer = AttentionVisualizer(
            self.model_manager,
            [self.prompt1, self.prompt2],
            self.demo_type
        )
        self.visualizer.visualize()

    def run(self):
        """Run the complete demo workflow."""
        self.execute_introduction()

        # Get first prompt
        self.prompt1 = self.get_user_prompt(1)
        self.prompt2 = self.get_user_prompt(2)

        print("\nYour two prompts are:")
        print(f"Prompt1: {self.prompt1}")
        print(f"Prompt2: {self.prompt2}")

        # Run model inference
        self.run_model_inference()

        # Launch visualization
=======
"""
Base class for attention probe demonstrations.

Refactor summary
----------------
* Added ``mode`` parameter to ``__init__``:
    - ``"legacy"``        – original single-prompt + auto-contrast flow.
    - ``"custom_pairs"``  – user supplies two fully freeform prompts (subject
                            pronoun variations, e.g. he/they, she/they).
    - ``"verb_violation"``– user supplies two freeform prompts to stress-test
                            grammatical violations (e.g. "he says" vs "he say").

* ``get_user_prompt()`` now routes on mode *and* on the config flag
  ``auto_generate_prompt2``.  When either ``custom_pairs`` or
  ``verb_violation`` is active **or** ``auto_generate_prompt2`` is False
  (which is the case for the refactored number_agreement config), the method
  bypasses all pronoun/keyword regex checks and enforces only a minimal
  string-length guard (> 5 characters) so completely freeform sentences are
  accepted without crashing.

* ``run()`` wiring updated to route both prompts through the freeform path
  when the comparative mode is active, while the legacy path is unchanged.
"""

from typing import List
from config import DEMO_CONFIGS, UI_CONFIG, COMMON_MESSAGES
from utils import ModelManager, validate_sentence, generate_contrast_prompt
from attention_visualizer import AttentionVisualizer

# Minimum character count accepted as a "valid" freeform prompt.
_MIN_PROMPT_LENGTH: int = 5


class BaseDemo:
    """Base class for all AttentionProbe demonstrations.

    Parameters
    ----------
    demo_type : str
        Key into ``DEMO_CONFIGS`` – one of ``"pronoun_resolution"``,
        ``"number_agreement"``, ``"noun_phrases"``, or
        ``"prep_phrase_attach"``.
    mode : str, optional
        Controls the prompt-ingestion strategy:

        ``"legacy"`` (default)
            Single-prompt entry with automatic contrast generation.  The
            original strict keyword validation is applied.

        ``"custom_pairs"``
            Two independent, unrestricted prompts are collected.  Intended
            for subject-pronoun variation experiments (he/they, she/they).

        ``"verb_violation"``
            Two independent, unrestricted prompts are collected.  Intended
            for grammatical-violation experiments (he says vs he say).
    """

    def __init__(self, demo_type: str, mode: str = "legacy") -> None:
        if demo_type not in DEMO_CONFIGS:
            raise ValueError(
                f"Unknown demo type: '{demo_type}'. "
                f"Valid options: {list(DEMO_CONFIGS.keys())}"
            )

        self.demo_type: str = demo_type
        self.mode: str = mode
        self.config: dict = DEMO_CONFIGS[demo_type]
        self.common_messages: dict = COMMON_MESSAGES
        self.model_manager: ModelManager = ModelManager("google/flan-t5-large")
        self.prompt1: str = ""
        self.prompt2: str = ""

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def execute_introduction(self) -> None:
        """Print the demo introduction, adapting the copy to the active mode."""
        print(f"\nHello! Welcome to {self.config['name']}.")

        mode_labels = {
            "legacy": (
                "In this demonstration, we will ask you to input one prompt and"
                " auto-generate the contrast swap."
            ),
            "custom_pairs": (
                "Mode: Multi-Prompt Expansion. Please enter 2 custom contrasting"
                " prompts (e.g., using he/they or she/they)."
            ),
            "verb_violation": (
                "Mode: Grammatical Violation Test. Enter pairs to evaluate syntax"
                " breakdowns (e.g., 'he says' vs 'he say')."
            ),
        }
        print(mode_labels.get(self.mode, ""))

        print(
            "We will run your sentences through Google's FLAN-T5-Large model"
            " and show you interesting attention heads.\n"
        )
        print(self.config["description"])
        print(self.config["description-details"])
        print(self.common_messages["navigation_msg"])

    def transition_description(self) -> None:
        """Print the transition message before launching the visualiser."""
        print(
            "\nNow we will present some of the notable attention heads."
            " Press q to exit the demonstration."
        )

    # ------------------------------------------------------------------
    # Prompt ingestion
    # ------------------------------------------------------------------

    def _is_freeform_mode(self) -> bool:
        """Return True when strict keyword validation should be skipped.

        Freeform mode is active when:
        * ``self.mode`` is ``"custom_pairs"`` or ``"verb_violation"``, OR
        * the config has ``auto_generate_prompt2 = False`` (which signals that
          the demo is designed for manual two-prompt comparison and does not
          rely on keyword-swap auto-generation).
        """
        return (
            self.mode in ("custom_pairs", "verb_violation")
            or not self.config.get("auto_generate_prompt2", True)
        )

    def get_user_prompt(self, prompt_number: int = 1) -> str:
        """Collect and validate a single prompt from the user.

        Validation strategy
        -------------------
        * **Freeform mode** (``custom_pairs``, ``verb_violation``, or any demo
          with ``auto_generate_prompt2 = False``): only a minimum-length check
          is applied.  No keyword or regex constraints are enforced, so the
          user can type fully custom sentences such as
          "The boy says everything" and "They say everything".
        * **Legacy / strict mode**: the original keyword-presence and
          single-occurrence validation is applied via ``validate_sentence()``.

        Parameters
        ----------
        prompt_number : int
            1 or 2 – used only for the terminal display label.

        Returns
        -------
        str
            The validated prompt string.
        """
        print(f"\nPlease input prompt {prompt_number} below.")
        print(
            f"  (type 0 to load the default: "
            f"'{self.config[f'default_prompt{prompt_number}']}' )"
        )

        while True:
            raw = input(f"My prompt {prompt_number}: ").strip()

            # ── Default shortcut ──────────────────────────────────────
            if raw == "0":
                return self.config[f"default_prompt{prompt_number}"]

            # ── Freeform mode: minimal length guard only ──────────────
            if self._is_freeform_mode():
                if len(raw) > _MIN_PROMPT_LENGTH:
                    return raw
                print(
                    f"  ✗ Prompt too short (minimum {_MIN_PROMPT_LENGTH + 1} characters)."
                    "  Please write a complete sentence."
                )
                continue

            # ── Legacy strict validation ──────────────────────────────
            if validate_sentence(
                raw,
                self.config["keywords"],
                self.config["max_keyword_count"],
            ):
                return raw

            print(
                f"  ✗ Invalid prompt. {self.config['validation_message']}\n"
                "  Please re-enter a valid prompt."
            )

    # ------------------------------------------------------------------
    # Contrast-prompt generation (legacy path only)
    # ------------------------------------------------------------------

    def generate_contrasting_prompt(self, original_prompt: str) -> str:
        """Auto-generate a contrasting prompt by swapping the keyword token.

        Only called in ``"legacy"`` mode when ``auto_generate_prompt2`` is
        True.  Falls back to interactive user input if generation fails.

        Parameters
        ----------
        original_prompt : str
            The validated first prompt.

        Returns
        -------
        str
            The contrasting prompt string.
        """
        try:
            result = generate_contrast_prompt(
                original_prompt,
                self.config["keywords"],
                self.config["auto_generate_prompt2"],
            )
            if result == "":
                # Auto-generation disabled at the config level; ask the user.
                return self.get_user_prompt(2)
            return result
        except ValueError:
            print(
                "  ⚠ Auto-contrast generation failed; falling back to default prompt 2."
            )
            return self.config["default_prompt2"]

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    def run_model_inference(self) -> None:
        """Run both prompts through the model and print the outputs."""
        print("\nRunning both prompts through the model:\n")
        for i, prompt in enumerate([self.prompt1, self.prompt2], start=1):
            response = self.model_manager.generate_response(
                prompt,
                max_length=UI_CONFIG["max_generation_length"],
            )
            print(f"  Prompt {i}: {prompt}")
            print(f"  Output {i}: {response}\n")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def launch_visualization(self) -> None:
        """Instantiate and launch the attention visualiser."""
        visualizer = AttentionVisualizer(
            self.model_manager,
            [self.prompt1, self.prompt2],
            self.demo_type,
        )
        visualizer.visualize()

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Orchestrate the full demo pipeline.

        Ingestion routing
        -----------------
        * **Freeform modes** (``custom_pairs``, ``verb_violation``, or any demo
          where ``_is_freeform_mode()`` returns True, e.g. the refactored
          ``number_agreement`` config): collect both prompts independently with
          no keyword restrictions.
        * **Legacy mode** (``pronoun_resolution`` with default settings): collect
          prompt 1 with strict validation, then auto-generate prompt 2 via
          keyword swap (or ask the user if the demo requires two explicit inputs
          and ``auto_generate_prompt2`` is False).
        """
        self.execute_introduction()

        # ── Prompt ingestion ──────────────────────────────────────────────
        if self._is_freeform_mode():
            # Comparative / freeform path: two fully independent, unrestricted
            # prompts are collected sequentially.
            self.prompt1 = self.get_user_prompt(1)
            self.prompt2 = self.get_user_prompt(2)

        else:
            # Legacy path: strict single-keyword validation on prompt 1; prompt
            # 2 may be auto-generated or solicited from the user.
            self.prompt1 = self.get_user_prompt(1)

            if self.config["need_prompt2"]:
                if self.config["auto_generate_prompt2"]:
                    print(
                        "\nAuto-generating a contrast sequence from your prompt…"
                    )
                    self.prompt2 = self.generate_contrasting_prompt(self.prompt1)
                else:
                    # need_prompt2 = True but auto_generate = False → ask user.
                    self.prompt2 = self.get_user_prompt(2)
            else:
                # Demo uses the same prompt for both slots (e.g. noun_phrases).
                self.prompt2 = self.prompt1

        # ── Echo final prompts ────────────────────────────────────────────
        print("\n── Final prompt pair ──────────────────────────────────────")
        print(f"  Prompt 1 : {self.prompt1}")
        print(f"  Prompt 2 : {self.prompt2}")
        print("───────────────────────────────────────────────────────────\n")

        # ── Processing pipeline ───────────────────────────────────────────
        self.run_model_inference()
>>>>>>> Stashed changes
        self.transition_description()
        self.launch_visualization()