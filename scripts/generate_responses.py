from pathlib import Path
import argparse
from src.utils import load_model_configs
from src.response_generation import generate, generate_claude_responses, generate_gpt_responses

def run_vllm_inference(
        model_id: str, 
        hf_path: str, 
        prompts_path: str,
        data_path: str, 
        num_gpus: int, 
        model_type: str,
        num_gen: int,
        max_tokens: int):

    if "google" in hf_path:
        dtype="bfloat16"
    else:
        dtype="float16"

    generate(input=prompts_path,
             output=f"{data_path}/{model_id}.json",
             model=hf_path,
             num_dev=num_gpus,
             model_type=model_type,
             num_gen=num_gen,
             max_tokens=max_tokens,
             dtype=dtype
             )
    

def run_api_inference(
        model_id: str, 
        provider: str,
        prompts_path: str,
        data_path: str, 
        num_gen: int,
        max_tokens: int):
    if provider=="claude":
        generate_claude_responses(
            prompt_file=prompts_path,
            numgen=num_gen,
            model=model_id,
            maxtokens=max_tokens,
            out_file=f"{data_path}/{model_id}.json"
        )
    elif provider=="gpt":
        generate_gpt_responses(
            prompt_file=prompts_path,
            model=model_id,
            maxtokens=max_tokens,
            numgen=num_gen,
            out_file=f"{data_path}/{model_id}.json"
        )
    else:
        print(f"Unsupported model {model_id}")

def main(args):
    # collect models
    models_path = Path(args.config_path)
    if not models_path.exists():
        current_script_dir = Path(__file__).resolve().parent
        models_path = current_script_dir.parent / models_path

    model_repository = load_model_configs(models_path)
    
    print(f"Loaded {len(model_repository)} models from configuration.\n")

    for model_id, spec in model_repository.items():
        hf_path = spec.path
        
        print(f"Processing model: {model_id} (Role: {spec.role} | Type: {spec.type})")
        
        if spec.gpus is not None:
            # send to vLLM
            run_vllm_inference(
                model_id=model_id,
                hf_path=hf_path,
                prompts_path=args.prompts_path,
                data_path=args.data_path,
                num_gpus=spec.gpus,
                model_type=spec.type,
                num_gen=spec.numgen,
                max_tokens=spec.tokens
            )
        else:
            # API Model (gpus is null) -> Send to API handler
            run_api_inference(
                model_id=model_id, 
                provider=spec.prefix,
                data_path=args.data_path,
                prompts_path=args.prompts_path,
                num_gen=spec.numgen,
                max_tokens=spec.tokens
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating responses")
    parser.add_argument("--prompts_path", type=str, default="data/prompts.json", help="input file containing a list of prompts")
    parser.add_argument("--config_path", type=str, default="configs/target_models.yaml", help="input file containing a list of prompts")
    parser.add_argument("--data_path", type=str, default="data/responses", help="input file containing a list of prompts")

    args = parser.parse_args()

    main(args)