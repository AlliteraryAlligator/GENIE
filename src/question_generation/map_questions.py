import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
from src.utils import parse_chat_completion, read_df, save_df, check_overwrite_safety



class Feature(BaseModel):
    question: str
    reasoning: str
    feature: str

class FeatureMap(BaseModel):
    features: list[Feature]


def map_questions(questions, output, model="gpt-4.1-mini"):
    print("Mapping Questions...")
    
    """
    Maps filtered questions to the most suitable feature (those that were independently discovered using feature_discover/ or defined)

    Args:
        questions: Path to questions with validity labels
        output: Path to save questions with feature mappings
        model: Model used for mapping questions to feature
    """
    
    feature_mapping_prompt = """
        Given a question, do the following:

        (1) Select the feature it best represents as defined below. Select only one feature per question.
        Features: 
        1. Agent - the characters involved in the narrative and their attributes, goals, motivations, backstories, personalities and arcs
        2. Perspective - includes point of view and focalization
        3. Plot - the content of the story (plotline, themes, obstacles, tropes, topics)
        4. Setting - where and when the story takes place, what unique objects define the location
        5. Structure - the overall structure of the plot includes conflict, rising suspense, change of fortune and resolution
        6. Social Network - interactions and relationships that characters have with each other
        7. Style - the language used, tone, figurative devices employed, etc.

        If the question does not reflect any of the features well, denote 'None'

        Follow this format:
        Question:
        Reasoning:
        Therefore the feature is: <feature>

        Questions:
        %s
    """

    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without mapping questions.")
        return

    question_df = read_df(questions)
    groups = question_df[question_df['Relevance']==True].groupby('Prompt')
    for (prompt), group in tqdm(groups, total=len(groups)):
        q_chunk = group['Question'].to_list()
        try:
            response = parse_chat_completion(
                model=model,
                messages=[
                    {"role": "user", "content": feature_mapping_prompt%q_chunk}
                ],
                response_format=FeatureMap,
            )
            features = response.features
            for feature_map in features:
                question = feature_map.question
                feature = feature_map.feature
                question_df.loc[(question_df['Prompt']==prompt) & (question_df['Question']==question), 'Feature'] = feature

        except Exception as e:
            print(f"Failed to map questions: {e}")
            continue

    save_df(question_df, output)


