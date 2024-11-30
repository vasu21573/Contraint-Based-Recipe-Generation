import re
import json
import os
import math
import torch
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import  AutoTokenizer,AutoModelWithLMHead



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



class Ratatouile:
    def forward(self,ingredients):
        novelRecipeGenerated,_=startRatatouileModel(ingredients)

        return novelRecipeGenerated

    def get_novel_recipe(self,ingredients,print_full=False):
        ingredients=";".join(ingredients)
        text=self.forward(ingredients)

        if(print_full):
            print(text)


        # POST-PROCESS-START
        start = text.find("<RECIPE_START>")
        end = text.find("<RECIPE_END>", start) + len("<RECIPE_END>")
        recipe_text = text[start:end]
        title_start = recipe_text.find("<TITLE_START>") + len("<TITLE_START>")
        title_end = recipe_text.find("<TITLE_END>")
        title = recipe_text[title_start:title_end].strip()
        ingr_start = recipe_text.find("<INGR_START>") + len("<INGR_START>")
        ingr_end = recipe_text.find("<INGR_END>")
        ingredients = recipe_text[ingr_start:ingr_end].strip().split(" <NEXT_INGR> ")
        instr_start = recipe_text.find("<INSTR_START>") + len("<INSTR_START>")
        instr_end = recipe_text.find("<INSTR_END>")
        instructions = recipe_text[instr_start:instr_end].strip().split(" <NEXT_INSTR> ")
        # POST-PROCESS-END

        recipe_json = {
            "title": title,
            "ingredients": ingredients,
            "instructions": instructions
        }

        return recipe_json
        
        

        