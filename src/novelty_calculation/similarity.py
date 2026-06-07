from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
from src.utils import read_df, save_df, check_overwrite_safety

def similarity(
        input: str,
        output: str
):
    """
    Finds the similarity between answers (target v. population) using an LLM.

    Args:
        input: Path to pairs of answers
        output: Path to save similarity judgments
    """
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without computing similarity.")
        return

    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct",
        tensor_parallel_size=2,       
        dtype="float16",              
        max_model_len=4096,
        max_num_seqs=128,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
    )
    rubric = """Given a question and two answers, determine if the two answers are similar on a scale of 1-4. The rubric for scoring is as following:
    0: one or both of the answers are marked as completely unspecified, not applicable or 'None'. This includes cases where the question was not answered completely.

    1: the answers are completely different. They describe different entities, concepts or perspectives, with little or no overlap. The key details in the answers do not align and may even contradict each other.

    2: there is a slight overlap between the two answers. They share a broad theme or surface similarity, but the specifics diverge significantly. They may use different wording for related things, but the details are not interchangeable.

    3: there is a moderate similarity between the two answers. The answers address the same general idea, but with notable differences in scope, emphasis, or added detail. They overlap on core concepts but introduce distinct elements that make them partially different.

    4: the two responses are essentially the same. The answers are interchangeable, they describe the same characters, outcomes, or relationships in slightly different words. No meaningful difference in scope, emphasis, or detail. If substituted, they would convey nearly the same idea without much loss of meaning.


    Given the above rubric, score the similarity of the following pair of answers given their corresponding question on a scale of 1-4. Follow the format below and do not provide any extraneous text. 
    <user>
    Question:
    Answer 1:
    Answer 2:
    <assistant>
    Reasoning: (1 short sentence)
    Therefore the answer is:

    <user>
    %s
    """

    pair_prompt = """
    Question: %s
    Answer 1: %s
    Answer 2: %s
    <assistant>
    """
    
    df = read_df(input)

    exclude_ans = ['None', 'None.', 'Unspecified', 'Unspecified.', 'Not specified', 'Not applicable', 'N/A', 'Not applicable.']
    prompts = []
    rows = []
    
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row['Answer_1'] in exclude_ans:
            continue
        elif row['Answer_2'] in exclude_ans:
            continue
        try:
            prompt = pair_prompt % (row['Question'], row['Answer_1'], row['Answer_2'])
            chat_template = [
                {
                "role": "user",
                "content": rubric % prompt
                }
            ]
            prompts.append(chat_template)
            rows.append(index)
        except:
            print("ERROR", index)
    outputs = llm.chat(messages=prompts, sampling_params=sampling_params)

    for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="regex"):
        df.at[rows[i], 'SimilarityJudgment'] = output.outputs[0].text
        
        try:
            match = re.findall(r"Therefore the answer is:\s*(-?\d+)", output.outputs[0].text)[0]
            df.at[rows[i], 'Similarity'] = match
        except:
            print("Error: could not regex")
            
    save_df(df, output)