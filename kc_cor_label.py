import os
import json
import openai
from openai import OpenAI
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import ast
from collections import defaultdict
import re
from collections import Counter
from openai_api import *
from vllm import LLM, SamplingParams
    
def get_problem_kc(kc_file):
    kc_cnt = 0

    uniq_kcs = set()
    with open(kc_file, 'rb') as f:
        kc_problem_dict = json.load(f)

        for key, val in kc_problem_dict.items():
            kc_cnt += len(val)
            for kc in val:
                uniq_kcs.add(kc)

    uniq_kcs = list(uniq_kcs)
    print('No. of KCs:', len(uniq_kcs))
    print('Avg Kcs per problem:', kc_cnt / len(kc_problem_dict))
 
    kc_dict_res = {uniq_kcs[i]: i for i in range(len(uniq_kcs))}

    return kc_problem_dict, kc_dict_res

def read_data(file, kc_problem_dict):
    df = pd.read_pickle(file)

    df['Score'] = np.where(df["Score_x"] == 1, 1, 0)


    df.drop(columns=['Score_x', 'Score_y'], inplace=True)

    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)

    ## Map problem KCs to each data
    df['knowledge_component'] = df['prompt'].map(kc_problem_dict)

    return df



# df should be incorrect subset
def get_kc_correctness(df):
    problems = df['prompt'].tolist()
    codes = df['Code'].tolist()
    kcs = df['knowledge_component'].tolist()

    client = OpenAIClient(False)

    generation_kwargs = {"n": 1, "temperature": 0, "response_format": {"type": "json_object"}}

    prompt_example = """# Example 1
## Problem: 
Given an int array of any length, return a new array of its first 2 elements. If the array is smaller than length 2, use whatever elements are present.

## Code:
public int[] frontPiece(int[] nums) {
    int[] num = new int[2]; 
    for (int i = 0; i < num.length(); i++) { 
        num[i] = nums[i];
    }
    return num;
}

## Knowledge Components: 
["numerical comparison", "array index manipulation", "for loop", "array initialization", "conditional logic", "return statement"]

## Expected Output:
{
  "reasoning": [
    "Uses (i < ...), but num.length() is invalid syntax in Java (should be .length): applicable + incorrect.",
    "Indexes arrays without guarding for nums.length < 2; fails on small inputs: applicable + incorrect.",
    "Loop present, but uses num.length() (invalid syntax): applicable + incorrect.",
    "Initializes fixed size 2, violating requirement to return fewer elements for short inputs: applicable + incorrect.",
    "Conditional logic related code is not shown in the code.",
    "Returns the array with valid syntax matching method signature: correct use."
  ],
  "scores": [0, 0, 0, 0, 0, 1]
}

# Example 2
## Problem:
Given a string and int n, return a string made of n repetitions of the last n characters.

## Code:
public String repeatEnd(String str, int n) { 
    StringBuilder sb = new StringBuilder(n*n);
    String last = str.substring(str.length()-n);
    for(int i=0; i<n; i++) 
        sb.append(last);
    return sb.toString();
}

## Knowledge Components: 
["string construction techniques", "arithmetic operations", "while loop", "for loop", "substring extraction", "string length operations", "string concatenation"]

## Expected Output:
{
  "reasoning": [
    "Uses StringBuilder and toString() to correctly build output.",
    "Calculates capacity and substring index correctly.",
    "While-loop is not shown in the code.",
    "Iterates exactly n times using standard for-loop syntax: correct use.",
    "Computes last = str.substring(str.length() - n) correctly.",
    "Calls str.length() to compute start index; correct usage.",
    "Concatenation via stbuild.append(last) in loop is valid."
  ],
  "scores": [1, 1, 0, 1, 1, 1, 1]
}
"""


    prompts = []
    for p, c, kc in zip(problems, codes, kcs):
        prompt = ("Now follow the instructions in system message, analyze the following problem, student code and knowledge components.\n\n"
                f"## Problem\n{p}\n\n"
                f"## Student code:\n{c}\n\n"
                f"## The knowledge component list is:\n{kc}\n\n"
            )
        total_user_prompt = prompt_example + "\n" + prompt

        prompts.append(total_user_prompt)
             
        
    system_message = """You are an expert Java programming tutor. You are given a Java programming problem along with a student code submission and a list of Knowledge Components (KCs). Your task is to evaluate whether each knowledge components (KC) is correctly applied in a student's code submission to the programming problem.
For each knowledge component, assign exactly one label:
1 = The KC appears and is used correctly.
0 = The KC appears but is incorrectly or incompletely applied (syntax, logic, or runtime error) OR the KC is not shown from the code.
Rules:
- A KC should be labeled as correct only if syntax is valid, logic is appropriate, and this KC would function correctly even if other parts fail.
- Generate a one sentence justification for each KC correctness labeling.
- Make sure every knowledge component in the provided list is labeled. 
- Take the examples as reference.
OUTPUT FORMAT:
Your output MUST be valid JSON in this format:
{
  "reasoning": ["One sentence factual reason referencing code and rule for KC_1...", "..."],
  "scores": [1/0, ...]
}
"""

    output = client.get_responses(prompts, 'gpt-4o-mini', system_message, generation_kwargs, False)

    with open('kc_label_gen_sel_gpt.pkl', 'wb') as f:
        pickle.dump(output, f)

    return output

# single kc_labeling function call to deal with invalid generation
def kc_label_by_iteration(row):
    problem = row['prompt']
    code = row['Code']
    kcs = row['knowledge_component']

    system_message = """You are an expert Java programming tutor. You are given a Java programming problem along with a student code submission and a list of Knowledge Components (KCs). Your task is to evaluate whether each knowledge components (KC) is correctly applied in a student's code submission to the programming problem.
For each knowledge component, assign exactly one label:
1 = The KC appears and is used correctly.
0 = The KC appears but is incorrectly or incompletely applied (syntax, logic, or runtime error) OR the KC is not shown from the code.
Rules:
- A KC should be labeled as correct only if syntax is valid, logic is appropriate, and this KC would function correctly even if other parts fail.
- Generate a one sentence justification for each KC correctness labeling.
- Make sure every knowledge component in the provided list is labeled, no KC is missing. 
- Take the examples as reference.
OUTPUT FORMAT:
Your output MUST be valid JSON in this format:
{
  "reasoning": ["One sentence factual reason referencing code and rule for KC_1...", "..."],
  "scores": [1/0, ...]
}
"""

    prompt_example = """# Example 1
## Problem: 
Given an int array of any length, return a new array of its first 2 elements. If the array is smaller than length 2, use whatever elements are present.

## Code:
public int[] frontPiece(int[] nums) {
    int[] num = new int[2]; 
    for (int i = 0; i < num.length(); i++) { 
        num[i] = nums[i];
    }
    return num;
}

## Knowledge Components: 
["numerical comparison", "array index manipulation", "for loop", "array initialization", "conditional logic", "return statement"]

## Expected Output:
{
  "reasoning": [
    "Uses (i < ...), but num.length() is invalid syntax in Java (should be .length): applicable + incorrect.",
    "Indexes arrays without guarding for nums.length < 2; fails on small inputs: applicable + incorrect.",
    "Loop present, but uses num.length() (invalid syntax): applicable + incorrect.",
    "Initializes fixed size 2, violating requirement to return fewer elements for short inputs: applicable + incorrect.",
    "Conditional logic related code is not shown in the code.",
    "Returns the array with valid syntax matching method signature: correct use."
  ],
  "scores": [0, 0, 0, 0, 0, 1]
}

# Example 2
## Problem:
Given a string and int n, return a string made of n repetitions of the last n characters.

## Code:
public String repeatEnd(String str, int n) { 
    StringBuilder sb = new StringBuilder(n*n);
    String last = str.substring(str.length()-n);
    for(int i=0; i<n; i++) 
        sb.append(last);
    return sb.toString();
}

## Knowledge Components: 
["string construction techniques", "arithmetic operations", "while loop", "for loop", "substring extraction", "string length operations", "string concatenation"]

## Expected Output:
{
  "reasoning": [
    "Uses StringBuilder and toString() to correctly build output.",
    "Calculates capacity and substring index correctly.",
    "While-loop is not shown in the code.",
    "Iterates exactly n times using standard for-loop syntax: correct use.",
    "Computes last = str.substring(str.length() - n) correctly.",
    "Calls str.length() to compute start index; correct usage.",
    "Concatenation via stbuild.append(last) in loop is valid."
  ],
  "scores": [1, 1, 0, 1, 1, 1, 1]
}
"""

    prompt = ("Now follow the instructions in system message, analyze the following problem, student code and knowledge components.\n\n"
                f"## Problem\n{problem}\n\n"
                f"## Student code:\n{code}\n\n"
                f"## The knowledge component list is:\n{kcs}\n\n"
            )
    
    total_user_prompt = prompt_example + "\n" + prompt

    
    response = openai.chat.completions.create(
            model='gpt-4o',
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": total_user_prompt}
        ],
            temperature = 0,
            n=1
        )

    reply = response.choices[0].message.content.strip()
   
    try:
        reply_ls = json.loads(reply)
        labels = reply_ls['scores']
        if len(labels) != len(kcs):
            labels = []
        
        return labels

    except:
        print("Error in parsing JSON")
        return []

# Use open source model for KC correctnes labeling with vllm
def open_source_kc_labeling(df):
    # vLLM model loading
    model = LLM(model="Qwen/Qwen3-Coder-30B-A3B-Instruct", quantization="fp8", dtype="bfloat16")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")

    prompt_ls = []
    system_message = """You are an expert Java programming tutor. You are given a Java programming problem along with a student code submission and a list of Knowledge Components (KCs). Your task is to evaluate whether each knowledge components (KC) is correctly applied in a student's code submission to the programming problem.
For each knowledge component, assign exactly one label:
1 = The KC appears and is used correctly.
0 = The KC appears but is incorrectly or incompletely applied (syntax, logic, or runtime error) OR the KC is not shown from the code.
Rules:
- A KC should be labeled as correct only if syntax is valid, logic is appropriate, and this KC would function correctly even if other parts fail.
- Generate a one sentence justification for each KC correctness labeling.
- Make sure every knowledge component in the provided list is labeled, no KC is missing. 
- Take the examples as reference.
OUTPUT FORMAT:
Your output MUST be valid JSON in this format:
{
  "reasoning": ["One sentence factual reason referencing code and rule for KC_1...", "..."],
  "scores": [1/0, ...]
}
"""


    for index, row in df.iterrows():
        problem = row['prompt']
        code = row['Code']
        kcs = row['baseline_knowledge_component']

        # construct the prompt
        prompt_example = """# Example 1
## Problem: 
Given an int array of any length, return a new array of its first 2 elements. If the array is smaller than length 2, use whatever elements are present.

## Code:
public int[] frontPiece(int[] nums) {
    int[] num = new int[2]; 
    for (int i = 0; i < num.length(); i++) { 
        num[i] = nums[i];
    }
    return num;
}

## Knowledge Components: 
["numerical comparison", "array index manipulation", "for loop", "array initialization", "conditional logic", "return statement"]

## Expected Output:
{
  "reasoning": [
    "Uses (i < ...), but num.length() is invalid syntax in Java (should be .length): applicable + incorrect.",
    "Indexes arrays without guarding for nums.length < 2; fails on small inputs: applicable + incorrect.",
    "Loop present, but uses num.length() (invalid syntax): applicable + incorrect.",
    "Initializes fixed size 2, violating requirement to return fewer elements for short inputs: applicable + incorrect.",
    "Conditional logic related code is not shown in the code.",
    "Returns the array with valid syntax matching method signature: correct use."
  ],
  "scores": [0, 0, 0, 0, 0, 1]
}

# Example 2
## Problem:
Given a string and int n, return a string made of n repetitions of the last n characters.

## Code:
public String repeatEnd(String str, int n) { 
    StringBuilder sb = new StringBuilder(n*n);
    String last = str.substring(str.length()-n);
    for(int i=0; i<n; i++) 
        sb.append(last);
    return sb.toString();
}

## Knowledge Components: 
["string construction techniques", "arithmetic operations", "while loop", "for loop", "substring extraction", "string length operations", "string concatenation"]

## Expected Output:
{
  "reasoning": [
    "Uses StringBuilder and toString() to correctly build output.",
    "Calculates capacity and substring index correctly.",
    "While-loop is not shown in the code.",
    "Iterates exactly n times using standard for-loop syntax: correct use.",
    "Computes last = str.substring(str.length() - n) correctly.",
    "Calls str.length() to compute start index; correct usage.",
    "Concatenation via stbuild.append(last) in loop is valid."
  ],
  "scores": [1, 1, 0, 1, 1, 1, 1]
}
"""   


        prompt = ("Now follow the instructions in system message, analyze the following problem, student code and knowledge components.\n\n"
                    f"## Problem\n{problem}\n\n"
                    f"## Student code:\n{code}\n\n"
                    f"## The knowledge component list is:\n{kcs}"
                )
        
        total_user_prompt = prompt_example + "\n" + prompt
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": total_user_prompt}
        ]

        inp_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompt_ls.append(inp_prompt)
    
    # generate the code using the model
    preds = generate_kc_label(model, tokenizer, prompt_ls)
    
    return preds

def generate_kc_label(model, tokenizer, prompt_ls):
    preds = []

    batch_size = 16
    pbar = tqdm(total=len(prompt_ls), desc="inference")
    for i in range(0, len(prompt_ls), batch_size):
        batch_prompts = prompt_ls[i:i+batch_size]
        inputs = []
        
        sampling_params = SamplingParams(n=1, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, max_tokens=400)
        outputs = model.generate(batch_prompts, sampling_params=sampling_params)
        decoded_preds = []
        for output_i in outputs:
            for gen_i in output_i.outputs:
                generated_content = gen_i.text.strip()
                generated_content = generated_content.replace("```json", "").replace("```", "").strip()
                decoded_preds.append(generated_content)
     
        preds.extend(decoded_preds)
        pbar.update(len(batch_prompts))
       
    pbar.close()
    return preds


def select_kc(problem, solution_code, kc_list):
    system_message = """You are an expert programming tutor. Given a Java programming problem along with a student code submission and a list of Knowledge Components (KCs), your task is to select all knowledge components (KC) that are shown in the student's code submission.

Rules:
- 1. Carefully examine the student code and all provided KCs.
- 2. Reason explicitly whether each KC is demonstrated in the code and generate a one sentence justification for each KC selection.
- 3. Based on your reasoning, select all KCs that are clearly evidenced in the code.

OUTPUT FORMAT:
Your output MUST be valid JSON in this format:
{
  "reasoning": ["One sentence factual reason referencing code and existence of selected KC_1...", "..."],
  "KCs": [selected_KC_1, ...]
}
"""

    prompt = ("Now follow the instructions in system message, analyze the following problem, student code and knowledge components and select all related KCs to the code.\n\n"
                f"## Problem\n{problem}\n\n"
                f"## Student code:\n{solution_code}\n\n"
                f"## The knowledge component list is:\n{kc_list}"
            )
    
    response = openai.chat.completions.create(
            model='gpt-4o',
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
        ],
            temperature = 0,
            n=1
        )

    reply = response.choices[0].message.content.strip()
   
    try:
        reply_ls = json.loads(reply)
        labels = reply_ls['KCs']
        
        return labels

    except:
        print("Error in parsing JSON")
        return []



def kc_solution_mapping(sample_sol_dict, problem_kc_map_dict):
    kc_solution_map = defaultdict(dict)

    for key, val in sample_sol_dict.items():
        kc_i = problem_kc_map_dict[key]

        for sol_i in val:
            kc_selected_i = select_kc(key, sol_i, kc_i)

            kc_solution_map[key][sol_i] = kc_selected_i
    
    with open('aied_kc_solution_map.pkl', 'wb') as f:
        pickle.dump(kc_solution_map, f)
    
    return kc_solution_map


def embed_code(codes, model, tokenizer, device):
    all_embs = []
    for i in tqdm(range(0, len(codes), 128), desc="Embedding"):
        batch = codes[i : i + 128]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**enc) 

        last_hidden = out.last_hidden_state
        attn_mask = enc["attention_mask"].unsqueeze(-1)  

        # mean pooling over tokens where attention_mask == 1
        summed = (last_hidden * attn_mask).sum(dim=1)               
        counts = attn_mask.sum(dim=1).clamp(min=1)                  
        mean_pooled = summed / counts                              

        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        all_embs.append(mean_pooled.detach().cpu().numpy())

    return np.vstack(all_embs)

def convert_solution_to_df(sample_sol_dict):
    data = []
    for problem, solutions in sample_sol_dict.items():
        for sol in solutions:
            data.append({'prompt': problem, 'Code': sol})
    
    df = pd.DataFrame(data)
    return df

def build_solution_index(sol_df, model, tokenizer, device):
    sol_df = sol_df.copy()
    sol_emb = embed_code(sol_df["Code"].tolist(), model, tokenizer, device).astype(np.float32)
    sol_df["_row"] = np.arange(len(sol_df))
    sol_df["_emb_row"] = sol_df["_row"]

    sol_emb_by_problem = {}
    sol_meta_by_problem = {}

    for pid, g in sol_df.groupby("prompt", sort=False):
        idx = g["_emb_row"].to_numpy()
        sol_emb_by_problem[pid] = sol_emb[idx] 
        sol_meta_by_problem[pid] = g.drop(columns=["_row", "_emb_row"]).reset_index(drop=True)

    return sol_emb_by_problem, sol_meta_by_problem


def match_nearest_solution(df, model, tokenizer, device, sol_emb_by_problem, sol_meta_by_problem, code_col, problem_col="prompt"):
    out_df = df.copy()

    sub_emb = embed_code(out_df[code_col].tolist(), model, tokenizer, device).astype(np.float32)

    best_idx = np.full(len(out_df), -1, dtype=np.int32)
    best_sim = np.full(len(out_df), -np.inf, dtype=np.float32)

    for pid, rows in out_df.groupby(problem_col).groups.items():
        rows = np.array(list(rows), dtype=np.int64)

        E_sol = sol_emb_by_problem[pid]  
        E_sub = sub_emb[rows]             

        sims = E_sub @ E_sol.T            

        local_best_idx = sims.argmax(axis=1).astype(np.int32)
        local_best_sim = sims[np.arange(len(rows)), local_best_idx].astype(np.float32)

        best_idx[rows] = local_best_idx
        best_sim[rows] = local_best_sim

    out_df["best_solution_index"] = best_idx
    out_df["best_cosine"] = best_sim


    def lookup_solution_id(row):
        pid = row[problem_col]
        j = row["best_solution_index"]

        meta = sol_meta_by_problem[pid]
        sel_code = meta.iloc[int(j)]["Code"]

        return sel_code
    
    out_df["closest_solution_code"] = out_df.apply(lookup_solution_id, axis=1)
    return out_df



def find_closest_solution(df, sample_sol_dict):
    MODEL_NAME = "microsoft/codebert-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    sol_df = convert_solution_to_df(sample_sol_dict)

    sol_emb_by_problem, sol_meta_by_problem = build_solution_index(sol_df, model, tokenizer, device)

    df_updated = match_nearest_solution(df, model, tokenizer, device, sol_emb_by_problem, sol_meta_by_problem, 'last_code', problem_col="prompt")

    return df_updated


def main():
    ## Uncomment this part for context-aware Code-KC mapping to map the kc to each solution for each problem
    ## sample_sol_dict: key: problem statement, value: list of solution candidates
    # with open('aied_sample_sol_dict.json', 'r') as f:
    #     sample_sol_dict = json.load(f)

    ## problem_kc_map_dict: key: problem statement, value: list of all possible KCs at problem level
    # with open('KC_cor_problem_kc_map.pkl', 'rb') as f:
    #     problem_kc_map_dict = pickle.load(f) 

    # kc_solution_map = kc_solution_mapping(sample_sol_dict, problem_kc_map_dict)

    # # load dataset and process to get first and last code submissions
    # df = pd.read_pickle('data/dataset_time.pkl')
    # df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)

    # first_last_df = df.groupby(['SubjectID', 'ProblemID']).agg(
    #     prompt=('prompt', 'first'),
    #     first_timestamp=('ServerTimestamp', 'first'),
    #     first_code=('Code', 'first'),
    #     last_timestamp=('ServerTimestamp', 'last'),
    #     last_code=('Code', 'last')
    # ).reset_index()

    # first_last_df_closest = find_closest_solution(first_last_df, sample_sol_dict)
    # df = df.merge(first_last_df_closest[['SubjectID', 'ProblemID', 'first_code', 'closest_solution_code']], on=['SubjectID', 'ProblemID'], how='left')

    ## Mapping the knowledge components from the closest solution
    # df['knowledge_component'] = df.apply(lambda row: kc_solution_map[row['prompt']][row['closest_solution_code']], axis=1)

    # kc_path should be the path to load the kc file
    kc_problem_dict, kc_no_dict = get_problem_kc(kc_path)

    df = read_data('data/dataset_time.pkl', kc_problem_dict)

    incor_df = df[df['Score'] == 0].reset_index(drop=True)
    cor_df = df[df['Score'] == 1].reset_index(drop=True)

    cor_df['kc_labels'] = [[1]*len(kc) for kc in cor_df['knowledge_component'].tolist()]


    labels = get_kc_correctness(incor_df)

    # labels = open_source_kc_labeling(incor_df)

    kc_label_ls, track_ls = [], []
    inval_cnt = 0
    for ind in range(len(labels)):
        item = labels[ind]
        try:
            item_json = json.loads(item)

            kc_label_ls.append(item_json['scores'])
        except:
            # print(f"Error in parsing JSON for index {ind}")
            inval_cnt += 1
            track_ls.append(ind)
            kc_label_ls.append([])
    
    print(f"Number of invalid JSON entries: {inval_cnt}")

    incor_df['gen_kc_labels'] = kc_label_ls
    final_df = pd.concat([cor_df, incor_df], ignore_index=True)


if __name__ == "__main__":
    main()
