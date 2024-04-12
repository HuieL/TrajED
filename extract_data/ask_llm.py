from data_utils import chat_gpt

import pathlib
from copy import deepcopy
from typing import List, Optional, Type, TypeVar
from dataclasses import dataclass


#path = r"E:\Data\aaai2024-outlierpaper\social-outliers\checkin-nola\Checkin.tsv"
path = r"E:\Data\aaai2024-outlierpaper\hunger-outliers\checkin-nola\checkin-nola.tsv"
df = pd.read_csv(path, sep = "\t")
n_neighbors = 1

#Work or Social: Green, Yellow, Red
truth = [[66, 809, 976, 4, 84, 268, 858, 416, 307, 956], [83, 478, 1, 244, 379, 161, 147, 353, 517, 364], [546, 644, 347, 62, 551, 992, 554, 949, 900, 57]]
#Hunger:
truth = [[351, 331, 871, 739, 347, 821, 63, 986, 951, 947, 769, 761, 548, 886, 643, 50, 467, 228, 798, 753, 511, 8, 385, 400, 836, 930, 766, 872, 435, 787], [895, 697, 491, 143, 235, 412, 927, 913, 271, 282, 275, 711, 957, 332, 914, 485, 863, 598, 985, 276, 805, 764, 556, 159, 377, 993, 604, 164, 288, 433], [144, 345, 989, 14, 361, 153, 305, 774, 535, 646, 622, 860, 680, 39, 336, 430, 885, 185, 232, 62, 18, 152, 692, 878, 461, 206, 5, 508, 219, 896]]

#dx = pd.read_csv(r"C:\Users\HuieL\VScodes\TrajectoryDistiallation\datasets\work\gpt4_1106_outputs.csv", sep=",")
all_ids = truth[0] + truth[1] + truth[2]
all_ = [i for i in range(1000)]
normals = [i for i in all_ if i not in all_ids]
normal_ids = random.sample(normals, 120)
sampling_ids = all_ids + normal_ids
print(sampling_ids)


text_trajectories= []
prompts = []
for id in sampling_ids:
    trajectory = text_trajectory(df, id)
    text_trajectories.append(trajectory)
    n = 0
    neighbor_ids = []
    while n < n_neighbors:
        c_id = random.randint(0, 999)
        if c_id not in all_ids and c_id!=id: neighbor_ids.append(c_id)
        n+=1
    
    comparison_trajectories = [text_trajectory(df, user) for user in neighbor_ids]
    text_one = get_prompt(Item(trajectory, comparison_trajectories), task_name = "prompt")
    prompts.append(text_one)

output_gpt_4, output_claude_2, anomaly_score = [0 for _ in range(len(all_ids))], [0 for _ in range(len(all_ids))], [0 for _ in range(len(all_ids))]
agg_data = zip_longest(*[sampling_ids, text_trajectories, prompts, output_gpt_4, anomaly_score], fillvalue = '')
with open(os.path.join(r".\dataset\hunger\prompts.csv"), 'w', encoding="UTF-8", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(['userid', 'text_trajectory', 'prompt', 'output', 'anomaly_score'])
    wr.writerows(agg_data)
myfile.close()

