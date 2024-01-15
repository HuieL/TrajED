import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, #Problem here
    TrainingArguments,
    AutoModelForCausalLM,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from tqdm import tqdm

import pandas as pd
import numpy as np
import ipdb
import transformers
from datetime import datetime

import pathlib
from copy import deepcopy
from typing import List, Optional, Type, TypeVar
from dataclasses import dataclass
import heapq
import csv
from itertools import zip_longest

from data_utils import chat_gpt
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = "hunger"
data_files = f"./datasets/{dataset_name}/pair.jsonl"

#DataFrame: {"input": ..., "output": ...}
train_dataset = load_dataset('json', data_files=data_files, split='train')  

#################################################Tokenize define:

def formatting_func(example):
    text = f"### Explanation: {example['output']}"
    return text

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=8192,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

base_model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    device_map="cuda")
model.half()

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True, 
)

tokenizer.pad_token = tokenizer.eos_token
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)

#################################################FineTune model using Lora:

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

#################################################Begin Training:
project = "trajectory-finetune"
base_model_name = "llama2-7b-hf"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_train_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        optim="adamw_hf",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        #report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    #data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
