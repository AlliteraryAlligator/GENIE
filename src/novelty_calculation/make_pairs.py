import pandas as pd
from src.utils import read_df, save_df, check_overwrite_safety
from datasets import load_dataset

def make_pairs(
        target_path: str,
        id_col: str,
        prompt_col: str,
        output: str,
        custom_population=True,
        population_path: str = None
):
    """
    Forms and saves similarity pairs between target and population answers (per prompt, per question).

    Args:
        population_path: Path to dataframe of population response answers (sampled population)
        target_path: Path to dataframe of target response answers
        id_col: Name of column containing the response ID
        prompt_col: Name of column containing the prompt
        output: Path to save similarity pairs (pre-similarity computation)
    """
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without making pairs.")
        return
    
    if custom_population and population_path:
        population = read_df(population_path)
    else:
        population = load_dataset("AlliteraryAlligator/sample-population")['train'].to_pandas()

    target = read_df(target_path)
    
    population[id_col] = population[id_col].astype(str)
    target[id_col] = target[id_col].astype(str)
    match_cols = [prompt_col, id_col, 'Model']

    # Sanity check: target documents should not be in the population
    df_merged = target.merge(population[match_cols], on=match_cols, how='left', indicator=True)
    target = df_merged[df_merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # create pairs
    similarity_pairs = pd.merge(
        target, 
        population, 
        on=['Question', prompt_col], 
        suffixes=('_1', '_2')
    )

    # reorg naming conventions/column order
    column_selection = [
        prompt_col, 'Question', 'Feature_1', f'{id_col}_1', 'Model_1', 'Answer_1',
        f'{id_col}_2', 'Model_2', 'Answer_2',
    ]
    similarity_pairs = similarity_pairs[column_selection]
    similarity_pairs = similarity_pairs.rename(columns={'Feature_1': 'Feature'})

    save_df(similarity_pairs, output)
