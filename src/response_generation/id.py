import pandas as pd
import numpy as np
from src.utils import read_df, save_df, check_overwrite_safety
ids = np.arange(1000000, 9999999)
np.random.shuffle(ids)

def id_responses(
        input: str,
        output: str,
        reference_path: str,
        reference_id_col='ID'
):
    """
    Assigns random IDs to documents
    
    Args:
        input: Dataframe of documents
        reference_path: Existing IDs to prevent collision
        reference_id_col: 'ID'
        output: Path to save documents with ID
    """
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without assigning IDs to responses.")
        return

    df = read_df(input)
    if reference_path:
        reference = read_df(reference_path)
        existing_ids = reference[reference_id_col].unique()
        valid_ids = ids[~np.isin(ids, existing_ids)]
    else:
        valid_ids = ids
    df['ID'] = valid_ids[:len(df)]

    save_df(df, output)