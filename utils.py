"""
Utility functions for the AttentionProbe application.
"""

from typing import List, Tuple, Optional
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import re


def find_difference(tokens1: List[str], tokens2: List[str]) -> int:
    """
    Find the index where two token lists differ.
    
    Args:
        tokens1: First list of tokens
        tokens2: Second list of tokens
        
    Returns:
        Index where tokens differ, or -1 if they are identical
    """
    if len(tokens1) != len(tokens2):
        raise ValueError("Token lists must have the same length")
    
    for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
        if t1 != t2:
            return i
    return -1


#def validate_sentence(sentence: str, required_keywords: List[str], max_count: int) -> bool:
    """
    Validate that a sentence contains exactly one of the required keywords.
    
    Args:
        sentence: Input sentence to validate
        required_keywords: List of keywords that should be present
        max_count: Maximum number of keywords allowed
        
    Returns:
        True if sentence is valid, False otherwise
    """
    #sentence_lower = sentence.lower()
    #words = re.findall(r'\b\w+\b', sentence_lower)
    
    # Check if any required keyword is present
    #found_keywords = [keyword for keyword in required_keywords if keyword in words]
    
    #if not found_keywords:
        #return False

    # Only one of the keywords should be in a sentence
    #if len(found_keywords) > 1:
        #return False
    
    # Check that no more than max_count keywords are present
    #if max_count > 0:
        #total_count = sum(words.count(keyword) for keyword in required_keywords)
    #else: # max_count must be -1, which means unlimited keywords are allowed (ex: for noun phrases)
        #return True
    #return total_count <= max_count

def validate_sentence(sentence: str, required_keywords: List[str], max_count: int) -> bool:
    """
    Validate that a sentence contains exactly one occurrence of one of the required keywords.

    Args:
        sentence: Input sentence to validate
        required_keywords: List of keywords that should be present
        max_count: Maximum number of keyword occurrences allowed (use 1 for pronouns)

    Returns:
        True if sentence is valid, False otherwise
    """
    sentence_lower = sentence.lower()
    words = re.findall(r'\b\w+\b', sentence_lower)

    # Count total occurrences of all required keywords
    total_count = sum(words.count(keyword) for keyword in required_keywords)

    # Must contain exactly one occurrence if max_count is 1
    if max_count == 1:
        if total_count != 1:
            return False
    elif max_count > 1:
        if total_count > max_count:
            return False
    # max_count == -1 -> unlimited keywords are allowed (ex: for noun phrases)
    return True


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def generate_contrast_prompt(original_prompt: str, keywords: List[str], auto_generate: bool, pronoun_map: dict) -> str:
    """
    Generate a contrasting prompt by replacing one pronoun with another.
    
    Args:
        original_prompt: Original prompt containing a keywords
        keywords: List of keywords to swap between
        
    Returns:
        New prompt with keyword swapped
    """
    if auto_generate:
        words = re.findall(r'\b\w+\b|\W+', original_prompt)
    
    # find the first keyword in the sentence that exists in pronoun_map
        found_keyword = next((w for w in pronoun_map if w.lower() in [word.lower() for word in words]), None)
        if not found_keyword:
            raise ValueError(f"No keyword from {list(pronoun_map.keys())} found in prompt")
    
    # get the mapped opposite pronoun
        other_keyword = pronoun_map[found_keyword]
    
    # replace first occurrence
        for i, cur in enumerate(words):
            if re.fullmatch(r'\w+', cur) and cur.lower() == found_keyword.lower():
                words[i] = other_keyword
                break

        return ''.join(words)
    else:
        return ""



class ModelManager:
    """Manages T5 model loading and inference."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.config = None
        
    def load_model(self):
        """Load the T5 model and tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy = False)
            print("tokenizer = ", repr(self.tokenizer)[0:80])
            #print("tokenizer = ", repr(self.tokenizer))
        if self.model is None:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        if self.config is None:
            self.config = T5Config.from_pretrained(self.model_name)
            
    def generate_response(self, prompt: str, max_length: int = 20) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated response
            
        Returns:
            Generated response text
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()
            
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0][1:-1])
    
    def get_attention_outputs(self, prompts: List[str], max_length: int = 20):
        """
        Generate attention outputs for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum length of generated response
            
        Returns:
            Model outputs with attention information
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)
        
        return self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_attentions=True,
            return_dict_in_generate=True,
            max_length=max_length
        ) 