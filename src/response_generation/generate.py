from vllm import LLM, SamplingParams
import pandas as pd
import argparse
from src.utils import read_df, save_df, check_overwrite_safety

def generate(input: str,
             output: str,
             model: str,
             num_dev: int,
             model_type: str,
             num_gen = 5,
             max_tokens = 750,
             dtype = "float16"
             ):
    """
    Args:
        input (str): path to dataframe of prompts
        output (str): path to response destination
        model (str): model used for inference
        num_dev (int): number of devices
        type (str): Instruct/Base; used for determining chat template
        num_gen (int): number of responses sampled
        max_tokens (int)
        dtype (str)
    """
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without generating responses.")
        return

    llm = LLM(
        model=model,
        tensor_parallel_size=num_dev, 
        dtype=dtype,           
        max_model_len=4096,  
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    prompts = read_df(input)
    num_generations = num_gen
    queries = []
    indices = []
    for index, row in prompts.iterrows():
        if "Write a story" in row['Prompt']:
            prompt = row['Prompt']
        else:
            prompt = "Write a story where: "+row['Prompt']
        
        for _ in range(num_generations):
            if model_type=="Base":
                queries.append(prompt)
            else:
                chat_template = [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ]
                queries.append(chat_template)
            indices.append(index)
    if model_type=="Base":
        outputs = llm.generate(queries, sampling_params=sampling_params)
    else:
        outputs = llm.chat(messages=queries, sampling_params=sampling_params)

    responses = []
    for i, output in enumerate(outputs):
        doc = output.outputs[0].text
        responses.append({
            "Prompt": prompts.at[indices[i],'Prompt'],
            "Model": model,
            "Document": doc,
            "MaxTokens": max_tokens
        })

    out_df = pd.DataFrame(responses)
    save_df(out_df, output)

