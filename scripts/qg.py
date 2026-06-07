# truncate - optional
# quality - optional
# id

import argparse
from src.question_generation import generate_questions, filter_questions, map_questions

def main(args):
    """
    Args:
        documents: path to input dataframe of processed and id-ed documents
        questions: path to dataframe of questions
        answers_output: path to save answers
    """

    generate_questions(
        prompts_file=args.prompts, 
        model=args.qg_model,
        output=args.qg_output
        )
    
    filter_questions(
        questions=args.qg_output,
        model=args.filtering_model,
        output=args.qg_output
    )

    map_questions(
        questions=args.qg_output,
        output=args.qg_output,
        model=args.mapping_model)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating responses")
    parser.add_argument("--prompts", type=str, required=True, help="input file containing the prompts")
    parser.add_argument("--qg_model", type=str, default="gpt-4.1-mini", help="model to use to generate questions")
    parser.add_argument("--filtering_model", type=str, default="gpt-4.1-mini", help="model to use to filter questions")
    parser.add_argument("--qg_output", type=str, required=True, help="path to save generated questions")
    parser.add_argument("--mapping_model", type=str, default="gpt-4.1-mini", help="model to use to map questions")

    args = parser.parse_args()

    main(args)