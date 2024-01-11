#Dataset Loader from the entire dataset
from datasets import DatasetDict
import heapq
import random

def TrajDatasetLoader(datasets, abnormal_samples, normal_samples, llm_label = True):
    train_dataset, test_dataset = list(), list()
    scores = [datasets[i]["anomaly_score"] for i in range(len(datasets))]

    if llm_label: 
        threld = heapq.nlargest(abnormal_samples, scores)
        normal_ids = [datasets[i]["id"] for i in range(len(datasets)) if datasets[i]["llm_label"] == "normal"]
        abnormal_sample_ids = [datasets[i]["id"] for i in range(len(datasets)) if scores[i] >= threld][:abnormal_samples]
        normal_sample_ids = random.sample(normal_ids, normal_samples)
    else: 
        normal_ids = [datasets[i]["id"] for i in range(len(datasets)) if datasets[i]["label"] == "normal"]
        abnormal_sample_ids = [datasets[i]["id"] for i in range(len(datasets)) if datasets[i]["label"] == "abnormal"]
        normal_sample_ids = random.sample(normal_ids, normal_samples)
    
    train_dataset = [datasets[i] for i in range(len(datasets)) if datasets[i]["id"] in abnormal_sample_ids or datasets[i]["id"] in normal_sample_ids]
    test_dataset = [datasets[i] for i in range(len(datasets)) if datasets[i] not in train_dataset]

    return  DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
