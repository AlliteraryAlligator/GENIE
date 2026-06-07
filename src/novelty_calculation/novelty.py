from src.utils import read_df, save_df, check_overwrite_safety


def compute_genie(
        input: str,
        genie_output: str,
        q_genie_output: str
):
    """
    GENIE and Q_GENIE computation: average dissimilarity between a target response and the population responses for the same prompt
    Args:
        input: input file containing a list of similarity pairs
        genie_output: output file name to save GENIE values
        q_genie_output: output file name to save Q GENIE values
    """
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(genie_output) or not check_overwrite_safety(q_genie_output):
        print("Exiting without computing genie values.")
        return

    sim_df = read_df(input)
    sim_df = sim_df.loc[sim_df['Similarity'].isin([1,2,3,4])] 
    sim_df_filtered = sim_df.copy()

    # normalize the similarity values (0-1 range)
    sim_df_filtered['Normalized'] = (sim_df_filtered['Similarity']-1)/3

    
    # 1. Q_GENIE
    q_novelty = (1 - sim_df_filtered.groupby(['Prompt', 'DocID_1', 'Model_1', 'Question', 'Answer_1', 'Feature'])['Normalized'].mean()).reset_index()

    # 2. GENIE
    novelty = q_novelty.groupby(['Prompt', 'DocID_1', 'Model_1', 'Feature'])['Normalized'].mean().reset_index()

    novelty = novelty.rename(columns={"Normalized":"GENIE"})
    save_df(novelty, genie_output)

    q_novelty = q_novelty.rename(columns={"Normalized":"Q_GENIE"})
    save_df(q_novelty, q_genie_output)

