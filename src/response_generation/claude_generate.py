"""
generate_docs.py

Run inference on a Claude model to generate documents from prompts.

Usage:
    python generate_docs.py \
        --model claude-sonnet-4-6 \
        --prompt_file prompts.json \
        --numgen 5 \
        --maxtokens 750 \
        --out_file outputs.json
"""
import os
import sys

import anthropic
import pandas as pd
from tqdm import tqdm
from src.utils import read_df, save_df, check_overwrite_safety

def load_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY environment variable not set.\n"
            "Please export your key before running:\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)

def generate_document(
        client: anthropic.Anthropic, 
        model: str, 
        prompt: str, 
        max_tokens: int) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def generate_claude_responses(
        prompt_file: str,
        numgen: int,
        model: str,
        maxtokens: int,
        out_file: str):
    
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(out_file):
        print("Exiting without generating responses.")
        return

    client = load_client()
    prompts_df = read_df(prompt_file)

    responses = []

    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
        if "Write a story" in row['Prompt']:
            prompt = row['Prompt']
        else:
            prompt = "Write a story where: "+row['Prompt']
            
        for gen_idx in range(numgen):
            try:
                document = generate_document(client, model, prompt, maxtokens)
            except anthropic.APIConnectionError as e:
                print(f"  Connection error: {e}", file=sys.stderr)
                document = None
            except anthropic.RateLimitError as e:
                print(f"  Rate limit error: {e}", file=sys.stderr)
                document = None
            except anthropic.APIStatusError as e:
                print(f"  API error {e.status_code}: {e.message}", file=sys.stderr)
                document = None

            responses.append({
                "Prompt":    prompt,
                "Model":     model,
                "Document":  document,
                "MaxTokens": maxtokens,
            })

    out_df = pd.DataFrame(responses)
    save_df(out_df, out_file)
