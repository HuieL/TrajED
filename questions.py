import numpy as np
import pandas as pd
import pickle as pkl
from itertools import zip_longest
import csv
from prompting import (
    Item,
    get_prompt
)

#Json keys: q_id, u_id, question, explaination, label
#Data: type, coor, dayofweek

trajectories, u_id = [], []
with open(r".\dataset\geolife\outliers-train-test\combined_prompt_with_hint.txt", 'r') as f:
    files = f.readlines()
    for line in files:
        if 'Here' in line.split():
            u_id.append(int(line.replace("\n", "").split(":",1)[0].split(" ")[-2]))
            trajectories.append(line.replace("\n", "").split(":",1)[1])

with open(".\dataset\geolife\outliers-train-test\groundtruth.txt", 'r') as f:
    files = f.readlines()
    for line in files:
        target = [int(i) for i in line.split(":")[1].replace("[", "").replace("]", "").replace(",", "").split(" ") if i != ""]

target = [i for i in target if i in u_id]
normal = [i for i in u_id if i not in target]

combine_prompts, record_id, explainations, scores, questions = [], [], [], [], []

for i in range(len(target)):
    t_i = trajectories[u_id.index(target[i])]
    compare_record, n = [], 0
    while n < 1:
        sample_id = np.random.randint(len(normal), size = 9)
        sample_id = [u_id[sample_id[x]] for x in range(len(sample_id))]
        ids = [u_id.index(j) for j in sample_id]
        if ids in compare_record: continue
        compare_record.append(ids)
        comparisons = [trajectories[i] for i in ids]
        combine = Item(t_i, comparisons)
        prompt_i = get_prompt(combine, task_name = "prompt")
        question_i = get_prompt(combine, task_name = "question")
        questions.append(question_i)
        combine_prompts.append(prompt_i)
        record_id.append(target[i])
        n+=1

for i in range(len(normal)):
    n_i = trajectories[u_id.index(normal[i])]
    compare_record, n = [], 0
    while n < 1:
        sample_id = np.random.randint(len(normal), size = 9)
        sample_id = [u_id[sample_id[x]] for x in range(len(sample_id))]
        ids = [u_id.index(j) for j in sample_id]
        if ids in compare_record and normal[i] in ids: continue
        compare_record.append(ids)
        comparisons = [trajectories[i] for i in ids]
        combine = Item(t_i, comparisons)
        prompt_i = get_prompt(combine, task_name = "prompt")
        question_i = get_prompt(combine, task_name = "question")
        questions.append(question_i)
        combine_prompts.append(prompt_i)
        record_id.append(normal[i])
        n+=1

# Example: geolife dataset
dataset = geolife_test
export_data = zip_longest(*[record_id, combine_prompts, questions, explainations, scores], fillvalue = '')
with open(f'.\datasets\prompts\{dataset}.csv', 'w', encoding="UTF-8", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(['user_id', 'prompts', 'question', 'explaination', 'score'])
      wr.writerows(export_data)
myfile.close()
