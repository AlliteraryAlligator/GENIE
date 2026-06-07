

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from src.utils import read_df, save_df, check_overwrite_safety

def score_coherence(
        responses: str,
        output: str,
        doc_col = "ProcessedDocument",
        prompt_col = "Prompt",
):
    """
    Score responses on coherence.

    Args:
        responses (str): Path to dataframe of responses to be scored
        output (str): Where to save scored responses 
        doc_col (str): Document column name
        prompt_col (str): Prompt column name
    """
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without computing coherence.")
        return
    
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

    model = AutoModelForSequenceClassification.from_pretrained(path, device_map="auto", 
                                   trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    df = read_df(responses)
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Scoring Coherence'):
        prompt = row[prompt_col]
        response = row[doc_col]
        messages = [{"role": "user", "content": prompt},
               {"role": "assistant", "content": response}]
        try:
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model(input_ids)
                multi_obj_rewards = output.rewards.cpu().float() 
        except:
            continue
        helpsteer_rewards_pred = multi_obj_rewards[0, :5] * 5 - 0.5
        df.loc[index, 'coherence'] = float(helpsteer_rewards_pred[2])
    
    df = df.dropna(subset=['coherence'])

    save_df(df, output)

