

import pandas as pd
from src.utils import read_df, save_df, check_overwrite_safety

sample_size = 50

def sample(docs, out_file):
    population_samples = []
    for (prompt), group in docs.groupby('Prompt'):
        population_samples.append(group.sample(min(sample_size, len(group)), random_state=39))

    sample_df = pd.concat(population_samples).reset_index(drop=True)
    
    save_df(sample_df, out_file)
    return sample_df

def sample_population(
        population_responses: str,
        population_answers: str,
        sampled_responses_output: str,
        sampled_response_answers_output: str
):
    """
    Creates the population by sampling at the document level (sample size is determined by a power analysis)
    population is sampled per prompt to ensure consistency when comparing target documents (genie values calculated against the same population)

    Args:
        population_ressponses: Path to population responses
        population_answers: Path to answers extracted from population responses
        sampled_responses_output: Path to save sampled population (responses)
        sampled_responses_answers_output: Path to save answers extracted from the sampled responses
    """

    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(sampled_response_answers_output) or not check_overwrite_safety(sampled_responses_output):
        print("Exiting without population sampling.")
        return

    docs_df = read_df(population_responses)
    docs_df = sample(docs_df, sampled_responses_output)
    doc_answers_df = read_df(population_answers)
    
    doc_answers_df = doc_answers_df.merge(
        docs_df[['ID', 'Prompt', 'Model']].rename(columns={'ID': 'DocID'}),
        on=['DocID', 'Prompt', 'Model'],
        how='inner'
    )
    
    save_df(doc_answers_df, sampled_response_answers_output)