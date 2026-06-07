from pydantic import BaseModel
from tqdm import tqdm
from src.utils import parse_chat_completion, save_df, read_df, check_overwrite_safety
import pandas as pd

class AspectQs(BaseModel):
    aspect: str
    subaspects: list[str]
    questions: list[str]

class PotentialQs(BaseModel):
    pq: list[AspectQs]


def generate_questions(prompts_file, output, model='gpt-4.1-mini'):
    print("Generating Questions...")
    """
    Generates questions specific to creative writing prompts that help extract the features of the task

    Args:
        prompts_file: Path to the list of prompts
        output: Path to save generated questions
        model: Model used for generating questions
    """

    qg_prompt = """
    You are an expert creative writing assistant. Your task is to help writers analyze and expand a creative writing prompt before they begin writing into a series of questions by following these steps. 
    You will be given a creative writing prompt. 

    Step 1: List the aspects of creative writing and how they might be relevant to the prompt. Focus on the construction of the writing product rather than specifics like world-building. Do not use complicated language or include extraneous details. Questions should be concise and precise, yet specific to the prompt.

    Step 2: For each aspect, describe the sub-aspects as they apply to the prompt. Aim for relevance and completeness. Then generate 5-10 questions that describe the sub-aspects as they apply to the prompt. These questions might help a writer while outlining.

    The questions must follow these rules:
    1. Questions should not be polar (yes/no) questions. 
    2. Examples must not be included in the question.  Incorrect: What fruits does the monkey like - apples, bananas or jack fruit? Correct: What fruits does the monkey like?
    3. A question can only ask one question at a time and may not use conjunctions for compounding. If there are multiple parts to the question, split them and ask separate questions. Incorrect: Who is the protagonist and what do they want? Correct: Who is the protagonist? What does the protagonist want?
    4. Avoid future tense or conditional verbs. 
    5. Questions are independent of each other and should not include anaphoric expressions.
    Prompt: %s
    """

    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without generating questions.")
        return
    
    prompts = read_df(prompts_file, lines=False)
    
    data = []
    for index, row in tqdm(prompts.iterrows(), total=len(prompts)):
        try:            
            response = parse_chat_completion(
                model=model,
                messages=[
                    {"role": "user", "content": qg_prompt%row['Prompt']}
                ],
                response_format=PotentialQs,
            )
            potential_questions = response.pq
            for aspect in potential_questions:
                for q in aspect.questions:
                    data.append({
                        "Prompt": row['Prompt'],
                        "Question": q,
                        "Aspect": aspect.aspect
                    })
        except Exception as e:
            print(f"Failed to generate questions: {e}")
            continue
    save_df(pd.DataFrame(data), output)


