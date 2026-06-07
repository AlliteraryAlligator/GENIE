import argparse
from src.novelty_calculation import compute_genie, sample_population, make_pairs, similarity

def main(args):

    if args.genie_population:
        # load and use the pre-existing sample population on huggingface
        if args.target_answers:
            make_pairs(
                target_path=args.target_answers,
                id_col='DocID',
                output=args.sim_pairs,
                custom_population=False
            )
    else:
        # build a custom population (i.e. sample from scratch)
        if args.population_responses and args.population_answers:
            sample_population(
                population_responses=args.population_responses,
                population_answers=args.population_answers,
                sampled_responses_output=args.sampled_population_responses,
                sampled_response_answers_output=args.sampled_population_answers
            )

        if args.sampled_population_answers and args.target_answers:
            make_pairs(
                population_path=args.sampled_population_answers,
                target_path=args.target_answers,
                id_col='DocID',
                prompt_col='Prompt',
                output=args.sim_pairs
            )


    if args.sim_pairs:
        similarity(
            input=args.sim_pairs,
            output=args.similarity
        )

    if args.similarity:
        compute_genie(
            input=args.similarity,
            genie_output=args.genie_output,
            q_genie_output=args.q_genie_output
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating responses")
    parser.add_argument("--use_pre_sampled_population", action='store_true', dest='genie_population', help="Use pre-sampled population dataset (default: False)")

    # sampling population
    parser.add_argument("--population_responses", type=str, help="Unsampled population responses")
    parser.add_argument("--population_answers", type=str, help="Path to answers to unsampled population responses")
    parser.add_argument("--sampled_population_responses", type=str, help="Path to sampled population (responses)")

    # make pairs
    parser.add_argument("--sampled_population_answers", type=str, help="Path to answers in sampled population responses")
    parser.add_argument("--target_answers", type=str, help="Path to answers in target responses")

    # similarity
    parser.add_argument("--sim_pairs", type=str, help="Path to save(d) similarity pairs between population and target answers")

    # genie
    parser.add_argument("--similarity", type=str, help="Path to save(d) similarity judgments between population and target answers")
    parser.add_argument("--genie_output", type=str, help="Path to save GENIE values")
    parser.add_argument("--q_genie_output", type=str, help="Path to save Q GENIE values")

    args = parser.parse_args()

    main(args)