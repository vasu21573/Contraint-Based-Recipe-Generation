import os
import math
import json
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelWithLMHead
from typing import List, Dict, Tuple



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def startRatatouileModel(ingredientsList):
  #Prepares model and provides the above random generated ingredients to Ratatouile model  
  MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
  }
  MODEL_CLASSES1 = {
    'gpt2': (AutoModelWithLMHead, AutoTokenizer),
  }
  model_class, tokenizer_class = MODEL_CLASSES['gpt2']
  tokenizer = tokenizer_class.from_pretrained('GPT2_NEW')
  model = model_class.from_pretrained('GPT2_NEW')
  model.to(torch.device("cuda" ))
  model.eval()

  raw_text=ingredientsList

  prepared_input = '<RECIPE_START> <INPUT_START> ' + raw_text.replace(',', ' <NEXT_INPUT> ').replace(';', ' <INPUT_END>')
  context_tokens = tokenizer.encode(prepared_input)

  out = sample_sequence(
    model=model,
    context=context_tokens,
    tokenizer=tokenizer,
    length=768,
    temperature=1, 
    top_k=30,
    top_p=1,
    device=torch.device("cuda")
  )
  out = out[0, len(context_tokens):].tolist()
  text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
  if "<RECIPE_END>" not in text:
    print(text)
    print("Failed to generate, recipe's too long")
  return text, prepared_input


def sample_sequence(model, length, context, tokenizer, num_samples=1, temperature=1, top_k=0, top_p=0.0, device = 'gpu'):
    end_token = tokenizer.convert_tokens_to_ids(["<END_RECIPE>"])[0]
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == end_token:
                print('breaking----->>')
                break
    return generated



class RatatouilleModel:
    def __init__(self, model_name: str = "GPT2_NEW", device: str = "cuda"):
        """
        Initializes the Ratatouille model.

        Args:
            model_name (str): Name of the pretrained model.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.device = torch.device(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_recipe(self, ingredients: List[str]) -> Dict[str, List[str]]:
        """
        Generates a novel recipe based on a list of ingredients.

        Args:
            ingredients (List[str]): List of input ingredients.

        Returns:
            Dict[str, List[str]]: Generated recipe including title, ingredients, and instructions.
        """
        raw_text = ";".join(ingredients)
        input_text = f"<RECIPE_START> <INPUT_START> {raw_text.replace(',', ' <NEXT_INPUT> ').replace(';', ' <INPUT_END>')}"
        context_tokens = self.tokenizer.encode(input_text)

        output_tokens = sample_sequence(
            model=self.model,
            tokenizer=self.tokenizer,
            context=context_tokens,
            length=768,
            temperature=1.0,
            top_k=30,
            top_p=0.9,
            device=self.device
        )
        generated_text = self.tokenizer.decode(output_tokens[0, len(context_tokens):], clean_up_tokenization_spaces=True)

        if "<RECIPE_END>" not in generated_text:
            raise ValueError("Failed to generate a complete recipe.")

        return self._post_process_recipe(generated_text)

    def _post_process_recipe(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts and structures recipe information from generated text.

        Args:
            text (str): Generated text from the model.

        Returns:
            Dict[str, List[str]]: Extracted recipe information.
        """
        recipe_json = {
            "title": self._extract_between(text, "<TITLE_START>", "<TITLE_END>"),
            "ingredients": self._extract_between(text, "<INGR_START>", "<INGR_END>").split(" <NEXT_INGR> "),
            "instructions": self._extract_between(text, "<INSTR_START>", "<INSTR_END>").split(" <NEXT_INSTR> ")
        }
        return recipe_json

    @staticmethod
    def _extract_between(text: str, start_tag: str, end_tag: str) -> str:
        """
        Extracts text between two tags.

        Args:
            text (str): Text to extract from.
            start_tag (str): Starting tag.
            end_tag (str): Ending tag.

        Returns:
            str: Extracted text.
        """
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        return text[start:end].strip()