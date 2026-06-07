from pydantic import BaseModel
from tqdm import tqdm
from src.utils import parse_chat_completion, read_df, save_df, check_overwrite_safety
import pandas as pd

class Filter(BaseModel):
    question: str
    reasoning: str
    relevance: bool

class FilterList(BaseModel):
    question_filter: list[Filter]

def filter_questions(questions, output, model='gpt-4.1-mini'):
    print("Filtering Questions...")

    """
    Determines the validity of questions based on criteria (see filter_prompt)

    Args:
        questions: Path to generated prompt-specific questions
        output: Path to save filtered questions
        model: Model to use to filter questions
    """

    filter_prompt = """
        Given a question, do the following:
        Decide if the question breaks any of the criteria below. If it does, mark it as irrelevant.
        1. Questions should not be polar (yes/no) questions. 
        2. Questions must not be speculative, i.e. they can be objectively and correctly answered with no subjectivity or analysis involved.
        3. Questions should not include intentions, hypotheticals, conditionals and should avoid the future tense.
        4. Questions must not be associated with multiple features (>=2) as defined below. Incorrect: How does the setting lend itself to imagery? This question belongs to both Setting and Style and is therefore irrelevant. 
        Features: 
        1. Agent - the characters involved in the narrative and their attributes, goals, motivations, backstories, personalities and arcs
        2. Perspective - includes point of view and focalization
        3. Plot - the content of the story (plotline, themes, obstacles, tropes, topics)
        4. Setting - where and when the story takes place, what unique objects define the location
        5. Structure - the overall structure of the plot includes conflict, rising suspense, change of fortune and resolution
        6. Social Network - interactions and relationships that characters have with each other
        7. Style - the language used, tone, figurative devices employed, etc.
        List all features that are clearly and fully applicable to the question. If there are more than one, reject the question.

        5. A question can only ask one question at a time and may not use conjunctions for compounding. Incorrect: Who is the protagonist and what do they want? Correct: Who is the protagonist? What does the protagonist want?
        6. Examples must not be included in the question.  Incorrect: What fruits does the monkey like - apples, bananas or jack fruit? Correct: What fruits does the monkey like?
        7. Questions are independent of each other and should not include anaphoric expressions. Incorrect: How is it resolved? Correct: How is the conflict between the main characters resolved?

        Follow this format:
        Question:
        Reasoning:
        Therefore the question is relevant: <True/False>

        Questions:
        %s
    """
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without filtering questions.")
        return

    question_df = read_df(questions)
    groups = question_df.groupby(['Prompt', 'Aspect'])
    for (prompt, aspect), group in tqdm(groups, total=len(groups)):
        q_chunk = group['Question'].to_list()
        try:
            response = parse_chat_completion(model=model,
                                             messages=[
                                                {"role": "user", "content": filter_prompt%q_chunk}
                                             ],
                                             response_format=FilterList)
            
            question_filters = response.question_filter
            for q_filter in question_filters:
                relevance = q_filter.relevance
                question = q_filter.question
                question_df.loc[(question_df['Prompt']==prompt) & (question_df['Question']==question), 'Relevance'] = relevance

        except Exception as e:
            print(f"Failed to filter questions: {e}")
            continue

    save_df(question_df, output)

