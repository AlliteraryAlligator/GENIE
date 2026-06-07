from vllm import LLM, SamplingParams
import re
from src.utils import read_df, save_df, check_overwrite_safety

qa_prompt = """Given the following document, answer these questions as succinctly as possible using as few words as possible. 
Decontextualize entities instead of using specific names (characters, places, etc.). 
Do NOT change the wording of the questions.

Example 1:
Who is the main character?
Correct: The baby whale
Incorrect: Aurora

Follow this format:
    Question: <the question>
    Answer: <the answer to the question>
    
    Question: <another question>
    Answer: <the answer to the question>

Task:
    Questions:\n%s
    Document:\n%s
"""

qa_long_prompt = """Given the following document, answer these questions. Decontextualize entities instead of using specific names (characters, places, etc.). 
Example 1:
Who is the main character?
Correct: The baby whale is the protagonist.
Incorrect: Aurora

Follow this format:
    Question: <the question>
    Answer: <the answer to the question>
    
    Question: <another question>
    Answer: <the answer to the question>

Task:
    Questions:\n%s
    Document:\n%s
"""
 


def answer(
        input: str, 
        questions: str, 
        output: str, 
        doc_column: str, 
        prompt_column: str, 
        id_column: str, 
        model_column: str,
        constrained: bool):

    """
    Extracts answers from responses

    Args:
        input: model-generated responses (target or population)
        questions: path to list of prompt-specific questions
        output: path to save answers
        doc_column: name of column to access the processed response
        prompt_column: name of column to access the prompt
        id_column: name of column to access the response id
        model_column: name of column to access the model that generated the response
        constrained: specifies which prompt to use to generate answers (e.g. constrained=True specifies concise answers)
    """
    
    # check if file exists/overwrite is allowed
    if not check_overwrite_safety(output):
        print("Exiting without answering questions.")
        return

    docs = read_df(input)
    question_df = read_df(questions)

    # conduct check to ensure docs has required columns
    
    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct",
        tensor_parallel_size=2,     
        dtype="float16",               
        max_model_len=16384,
        max_num_seqs=256,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1500,
    )
    queries = []
    indices = []

    if constrained:
        template = qa_prompt
    else:
        constrained = qa_long_prompt

    for index, row in docs.iterrows():
        questions = question_df[question_df['Prompt']==row[prompt_column]]['Question'].to_list()
        
        chat_template = [
                {
                "role": "user",
                "content": template % ("\n".join(questions), row[doc_column])
                }
            ]
        queries.append(chat_template)
        indices.append(index)
    outputs = llm.chat(messages=queries, sampling_params=sampling_params)
    
    qa_data = []
    for i, output in enumerate(outputs):
        qa_str = output.outputs[0].text
        

        try:
            prompt = docs.at[indices[i], prompt_column]

            pattern = re.compile(
                r"\n\s*Question\s*:\s*(.+?)\s*\n\s*Answer\s*:\s*(.+?)(?=\n\s*Question\s*:|\n\s*Answer\s*:|$)",
                re.DOTALL | re.IGNORECASE
            )

            # print(qa_str)
            # print(pattern.finditer(qa_str))

            for match in pattern.finditer(qa_str):
                q = match.group(1).strip()
                a = match.group(2).strip()
            
                try:    
                    qa_data.append({
                        "DocID": docs.at[indices[i], id_column],
                        "Question": q,
                        "Answer": a,
                        "Feature": question_df.loc[(question_df['Prompt']==prompt) & (question_df['Question']==q), 'Feature'].item(),
                        "Prompt": docs.at[indices[i], prompt_column],
                        "Model": docs.at[indices[i], model_column],
                    })
                except:
                    print("Individually printing items")
                    print(docs.at[indices[i], id_column])
                    print(q)
                    print(a)
                    print(question_df.loc[(question_df['Prompt']==prompt) & (question_df['Question']==q), 'Feature'].item())
                    print(docs.at[indices[i], prompt_column])
                    print(docs.at[indices[i], model_column])
        except:
            print("Error: could not regex")
            try:
                print(qa_str)
            except:
                print("Could not print pairs")

    save_df(qa_data, output)




