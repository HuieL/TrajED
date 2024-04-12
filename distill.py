import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig
from tqdm import tqdm

import pandas as pd
import numpy as np
import transformers
from datetime import datetime
import csv
from itertools import zip_longest
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = "geolife"
data_files = f"./datasets/{dataset_name}/pair_sup.jsonl"

#DataFrame: {"input": ..., "output": ...}
train_dataset = load_dataset('json', data_files=data_files, split='train')  

################################################# Tokenize define:

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

################################################# FineTune model using Lora:

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
base_model_name = "llama2-7b-chat"
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
        logging_dir="./logs",        
        save_strategy="steps",       
        save_steps=50,               
        evaluation_strategy="steps", 
        eval_steps=50,              
        do_eval=True,               
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          
    ),
    #data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

model.config.use_cache = False  # Re-enable for inference!
trainer.train()
