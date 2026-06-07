import pandas as pd
from tqdm import tqdm
import re
from pydantic import BaseModel
from src.utils import parse_chat_completion
from src.utils import read_df, save_df, check_overwrite_safety

truncation_prompt = """
Given a prompt and its response, perform the following task:

Purpose: find the contiguous chunk of the response that directly answers the prompt. Output the start and end sentence indices of that chunk, plus two booleans: Irrelevant and Incomplete. Watch out for signs of text degeneration. The ending sentence should exclude the point after which the text starts repeating, becomes incoherent or completely off topic.

Definitions:
Relevant content = any contiguous sequence of sentences in the response that directly and concretely addresses the user prompt (e.g., gives the requested story, factual answer, or explicit actions/outputs asked for). It should be content intended as the user's answer, not commentary about how the answer was created.
Irrelevant content = reasoning about methodology, planning, internal chain-of-thought, meta-commentary ("I will...", "Let's look at...", "Plan:", "Here’s how I'd approach this"), the assistant addressing the user about its process, or any content that does not itself deliver the requested answer/output. 
Incomplete = the response cuts off or clearly indicates more content is expected (ends mid-sentence, ends with "to be continued", or contains placeholders like "[...]" or "more to come").

Decision flow:
If the entire response is meta/reasoning (all sentences are Irrelevant by the above definition), set Irrelevant: true, Incomplete: false (unless it is cut off) and set Start line: none and End line: none.
If any contiguous chunk of sentences is a direct answer, set Irrelevant: false and mark Start/End to that chunk. If there are multiple discontiguous answer chunks, choose the largest contiguous chunk most directly answering the prompt. Ignore headings/labels (like "###") unless they are themselves the answer content.
If the response mixes meta and answer: exclude leading/trailing meta. Start at the first sentence that is actual answer content; end at the last sentence that still contributes to the answer. If there's ambiguity, prefer marking more conservatively (prefer smaller answer chunk).
If no part of the response answers the prompt, follow step 1.

Surface cues for meta/reasoning:
Phrases like "Let's look at...", "Let's plan", "Approach:", "My thinking:", "Here’s what changes", "I would...", "We should...", "Fun setup" — treat as meta unless immediately followed by content that is clearly the answer.
Bulleted lists that describe methods, planning, or analysis of differences rather than producing the requested narrative/output are meta.

Output format requirement:
If Irrelevant: true then Incomplete and Start/End must be "none" (unless Incomplete is true).
If Irrelevant: false then Start/End must be the exact first and last sentence (copy text) of the chosen answer chunk.
Incomplete: true if the response is cut off.

Provide your output like this:
Irrelevant: true/false
Incomplete: true/false
Start: start line
End: ending line

Instruction: %s
Response: %s

"""

class TruncatedResponse(BaseModel):
    irrelevant: bool
    incomplete: bool
    start: int
    end: int

def run_gpt(queries, indices, docs, model):
 
    for i, chat_template in enumerate(tqdm(queries, desc="GPT truncation")):
        try:
            truncated_response = parse_chat_completion(
                model=model,
                messages=chat_template,
                response_format=TruncatedResponse,
            )
            docs.at[indices[i], 'Irrelevant'] = truncated_response.irrelevant
            docs.at[indices[i], 'Incomplete'] = truncated_response.incomplete
            docs.at[indices[i], 'Start'] = truncated_response.start
            docs.at[indices[i], 'End'] = truncated_response.end
            
        except Exception as e:
            print(f"GPT call failed for index {indices[i]}: {e}")
            try:
                docs.at[indices[i], 'ProcessedDocument'] = docs.at[indices[i], 'Document']
            except Exception:
                print("could not update processed-document column")
 
    return docs
 

def truncate(
        input: str,
        output: str,
        truncation_model="gpt-4.1-mini"
):
    """
    Truncates responses to exclude incoherent, off-topic (output for other prompts), or reasoning text.
    
    Args:
        input: input file containing a list of responses
        output: path to save truncated responses
        truncation_model: model used for truncation (default = gpt-4.1-mini)
    """
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without truncating responses.")
        return
    
    docs = read_df(input)

    queries = []
    indices = []
    for index, row in docs.iterrows():
        chat_template = [
                {
                "role": "user",
                "content": truncation_prompt% (row['Prompt'], row['Document'])
                }
            ]
        queries.append(chat_template)
        indices.append(index)

    docs = run_gpt(queries, indices, docs, truncation_model)


    try:
        for index, row in docs.iterrows():
            if row['Irrelevant']:
                # response was completely off
                docs.at[index, 'ProcessedDocument'] = ""
            else:
                original_doc = row['Document']

                start_match = re.search(re.escape(row['Start']), original_doc)
                
                end_match = list(re.finditer(re.escape(row['End']), original_doc))
                
                if start_match is not None and end_match is not None and len(end_match)>0: 
                    content = original_doc[start_match.start():end_match[-1].end()]
                elif start_match is None:
                    if end_match is None or len(end_match)==0:
                        content = ""
                    else:
                        content = original_doc[:end_match[-1].end()]
                else:
                    content = original_doc[start_match.start():]

                docs.at[index, 'ProcessedDocument'] = content
    except:
        print("ERROR: could not chunk using start and end lines.")
    
    save_df(docs, output)
