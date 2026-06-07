"""
generate_docs.py

Run inference on a GPT model to generate documents from prompts.

Usage:
    python generate_docs.py \
        --model gpt-4.1 \
        --prompt_file prompts.json \
        --numgen 5 \
        --maxtokens 750 \
        --out_file outputs.json
"""

import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from src.utils import read_df, save_df, check_overwrite_safety
from typing import Optional

def generate_document(client, prompt: str, model_name: str, max_tokens: int):
    

    if model_name.startswith("gpt-5"):
        # e.g. gpt-5.1, gpt-5-mini
        response = client.responses.create(
            model=model_name,
            input=prompt,
        )
        text = response.output_text
        return text
    else:
        # e.g. gpt-4.1, gpt-4.1-mini, 
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        return text
        
    
def _build_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client, preferring the supplied key then env var."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No OpenAI API key found. Pass api_key= or set OPENAI_API_KEY."
        )
    return OpenAI(api_key=key)

def generate_gpt_responses(
        prompt_file: str,
        model: str,
        maxtokens: int,
        numgen: int,
        out_file: str):
    
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(out_file):
        print("Exiting without generating responses.")
        return

    client = _build_client()
    prompts_df = read_df(prompt_file)

    responses = []

    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
        prompt = row["Prompt"]
        for gen_idx in range(numgen):
            try:
                document = generate_document(client, prompt, model, maxtokens)
            except Exception as e:
                print(e)
                print("Could not generate document")
                document = None
            responses.append({
                "Prompt": prompt,
                "Model": model,
                "Document": document,
                "MaxTokens": maxtokens,
            })

    out_df = pd.DataFrame(responses)
    save_df(out_df, out_file)
