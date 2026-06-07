from pathlib import Path
import pandas as pd
import os
import sys

def read_df(path: str, orientation='records', lines=True):
    docs = Path(path).resolve() 
    if docs.is_file():
        doc_df = pd.read_json(docs, orient=orientation, lines=lines)
    else:
        dfs = []
        for root, directory, files in os.walk(docs):
            for entry in files:
                path = os.path.join(root, entry)
                try:
                    df = pd.read_json(path, orient=orientation, lines=lines)
                except:
                    print(f"Could not read {path}")
                    continue
                dfs.append(df)
        doc_df = pd.concat(dfs).reset_index(drop=True)

    return doc_df

def save_df(df: pd.DataFrame, path: str):
    output_file = Path(path).resolve()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_json(output_file, orient='records', lines=True)
    print(f"\n{len(df)} records saved to '{output_file}'.")

def check_overwrite_safety(path: str):
    output_file = Path(path).resolve()
    
    if output_file.exists():
        print(f"WARNING: The file '{output_file.name}' already exists.")
        print(f"Full path: {output_file}")
        
        # Prompt the user for input
        choice = input("Do you want to overwrite it? (y/N): ").strip().lower()
        
        if choice not in ['y', 'yes']:
            print("Operation cancelled by user. Exiting script to prevent overwrite.")
            return False
            
        print("Overwrite confirmed. Proceeding with computation...\n")
    return True
