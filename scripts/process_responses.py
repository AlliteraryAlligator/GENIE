# truncate - optional
# quality - optional
# id

import argparse
from src.response_generation import id_responses, score_coherence, truncate
from src.utils import read_df, save_df

def main(args):
    if args.truncate:
        truncate(
            input=args.input,
            output=args.output,
            truncation_model=args.truncation_model
        )
    else:
        # bypass truncation
        df = read_df(args.input)
        df['ProcessedDocument'] = df['Document']
        save_df(df, args.output)

    
    if args.quality:
        score_coherence(
            responses=args.output, 
            output=args.output
        )  # truncated output (or renamed col) exists in args.output
    else:
        df = read_df(args.output)
        df['coherence'] = 6     # always above quality threshold; effectively no quality filter
        save_df(df, args.output)


    id_responses(
        input=args.output, 
        output=args.output, 
        reference_id_col=args.reference_id_col, 
        reference_path=args.reference_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating responses")
    parser.add_argument("--input", type=str, required=True, help="input file containing raw model responses")
    parser.add_argument("--output", type=str, required=True, help="path to output")

    # truncation
    parser.add_argument("--no_truncate", action='store_false', dest='truncate', help="Do not truncate responses (default: True)")
    parser.add_argument("--truncation_model", type=str, default="gpt-4.1-mini", help="model used for truncation")

    # quality
    parser.add_argument("--no_quality_filter", action='store_false', dest='quality', help="Do not compute coherence for responses (default: True)")
    
    # id
    parser.add_argument("--reference_path", type=str, help="existing responses to prevent id collisions")
    parser.add_argument("--reference_id_col", type=str, help="name of ID column in reference response dataframe")

    args = parser.parse_args()

    main(args)