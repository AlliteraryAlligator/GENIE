# GENIE: A Fine-Grained Measure for Novelty
Authors: Ramya Namuduri, Manya Wadhwa, Anshun Asher Zheng, Greg Durrett, Junyi Jessy Li

GENIE finds the fine-grained novelty of LLM-generated responses with respect to a population defined by you! This repo currently demonstrates how to find the novelty of creative writing responses.

Check out the [paper](https://arxiv.org/abs/2606.12790) and [data](https://huggingface.co/collections/AlliteraryAlligator/genie)!

## Getting Started
Install dependencies:
```
pip install -r requirements.txt
```

To apply quality filtering (`src/response_generation/quality_filter.py`), use `transformers==4.39.0`.

Update the paths to the two virtual environments in `examples/`.

## Quickstart
Generate and map prompt-specific questions (prompts in `configs/prompts.json`):
```
./qg.sh
```

Compute GENIE:
```
./genie.sh
```

Discover features for a specific task:
```
./feature_discover.sh
```

To find features for a custom task:

```
python -m src.feature_discovery.discover_features  --task <custom-task> --out_file data/features/features.json
```

Note: the prompts used in other parts of the GENIE pipeline are specifically instantiated on the creative writing task using creative writing features.

## Additional Usage and Configuration
### Responses
Build a population and/or target pool of responses:
```
python -m scripts.generate_responses --prompts_path data/prompts.json \
    --config_path configs/models.yaml \
    --data_path data/responses
```

Truncate incoherent text, apply a quality filter and assign IDs to responses:
```
python -m scripts.process_responses --input <input file> --output <save file>
```

Optionally, use an existing dataset of responses and skip the generation step. Ensure that the response column is named `Document`. Then, truncate/filter/id the responses. To bypass truncation and filtering, and only assign IDs (required), use the `--no_truncate` and `--no_quality_filter` flags:

```
python -m scripts.process_responses --input <input file> --output <save file> --no_truncate --no_quality_filter
```

### Computing GENIE
#### Answers
Extract answers to questions generated previously from responses:
```
python -m scripts.answer --documents <processed responses> --questions <questions> --output <output path>
```
Pass in the `--unconstrained` flag for lengthier answers. By default, the answering prompt explicitly requires concise answers.

#### GENIE
Compute GENIE as described in `examples/genie.sh`. All arguments are paths to either save and/or read data. 

If using the entire population of responses, **without** sampling:
```
python -m scripts.genie 
    --sampled_population_answers <all population answers> 
    --target_answers <target answers> 
    --sim_pairs <similarity pairs>
    --similarity <similarity scores>
    --genie_output <genie scores>
    --q_genie_output <q_genie scores>
```

To use the population we built, instead of a custom population, set the `--use_pre_sampled_population` flag. This will load the `AlliteraryAlligator/sample-population` dataset.
```
python -m scripts.genie --use_pre_sampled_population
    --target_answers <target answers> 
    --sim_pairs <similarity pairs>
    --similarity <similarity scores>
    --genie_output <genie scores>
    --q_genie_output <q_genie scores>
```

Finally, if similarity was previously computed, run:
```
python -m scripts.genie 
    --similarity <similarity scores>
    --genie_output <genie scores>
    --q_genie_output <q_genie scores>
```
