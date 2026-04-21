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
        self.transition_description()
        self.launch_visualization() 