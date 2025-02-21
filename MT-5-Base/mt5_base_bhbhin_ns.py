import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()

os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
model_name = "google/mt5-base"
csv_file = "/home/pooja/Prince/MT-5/MT-5-base/train.csv"   
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)

def load_csv_dataset(file_path):
    df = pd.read_csv(file_path)
    #df = df.sample(5000, random_state=42).reset_index(drop=True)
    dataset = Dataset.from_pandas(df)
    train_test_split = dataset.train_test_split(test_size=100 if len(dataset) > 100 else 0.1)
    return DatasetDict({
        "train": train_test_split["train"],
        "validation": train_test_split["test"]
    })

dataset_splits = load_csv_dataset(csv_file)

def preprocess_function(examples):
    inputs = examples["Bhili"]
    targets = examples["Hindi"]

    inputs = [str(inp) if pd.notna(inp) else "" for inp in inputs]
    targets = [str(tgt) if pd.notna(tgt) else "" for tgt in targets]

    model_inputs = tokenizer(
        inputs, 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )

    labels = tokenizer(
        targets,
        max_length=128,
        padding="max_length",
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    
    model_inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in model_inputs["labels"]
    ]
    
    return model_inputs

tokenized_datasets = dataset_splits.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset_splits["train"].column_names
)
 
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-base-bhili-hi-finetuned",
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=8000,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=3,
    fp16=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
    predict_with_generate=True,
    generation_max_length=128,
    warmup_steps=500,
    gradient_checkpointing=True,
    optim="adamw_torch"
)
 
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)
 
trainer.train()
 
model.save_pretrained("./mt5-base-bhili-hi-finetuned")
tokenizer.save_pretrained("./mt5-base-bhili-hi-finetuned")

print("Fine-Tuning Complete! Model and tokenizer saved.")
