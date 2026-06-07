import argparse
from src.novelty_calculation import answer
from src.utils import read_df

def main(args):
    """
    Args:
        documents: path to input dataframe of processed and id-ed documents
        questions: path to dataframe of questions
        answers_output: path to save answers
    """

    responses = read_df(args.documents)

    if args.id_column not in responses.columns:
        # sanity check: responses must go through minimum pre-processing (id)
        print("Responses do not have assigned IDs, or there was an ID column mismatch.")
        return

    constrained_flag = True
    if not args.constrained:
        # user specified the unconstrained flag --> generate full answers
        constrained_flag = False

    answer(
        input=args.documents,
        questions=args.questions,
        output=args.output,
        doc_column=args.doc_column,
        prompt_column=args.prompt_column,
        id_column=args.id_column,
        model_column=args.model_column,
        constrained=constrained_flag
    )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating responses")
    parser.add_argument("--documents", type=str, required=True, help="input file containing processed model responses")
    parser.add_argument("--questions", type=str, required=True, help="path to questions")
    parser.add_argument("--output", type=str, required=True, help="path to output")
    parser.add_argument("--doc_column", type=str, default='ProcessedDocument')
    parser.add_argument("--id_column", type=str, default='ID')
    parser.add_argument("--model_column", type=str, default='Model')
    parser.add_argument("--prompt_column", type=str, default='Prompt')
    parser.add_argument("--unconstrained", action='store_false', dest='constrained', help="Constrain the length of responses (default: True)")

    args = parser.parse_args()

    main(args)