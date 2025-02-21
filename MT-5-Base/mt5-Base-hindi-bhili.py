import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
 
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
    inputs = examples["Hindi"]
    targets = examples["Bhili"]

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
 
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    try:
        predictions, labels = eval_pred
         
        if len(predictions.shape) == 3:
            predictions = np.argmax(predictions, axis=-1)
         
        vocab_size = tokenizer.vocab_size
        predictions = np.clip(predictions, 0, vocab_size - 1)
        
        predictions = predictions.tolist()
        labels = labels.tolist()
      
        decoded_preds = []
        decoded_labels = []
        
        for pred in predictions:
            try: 
                pred = [p for p in pred if 0 <= p < vocab_size]
                decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
                decoded_preds.append(decoded_pred)
            except Exception as e:
                print(f"Error decoding prediction: {e}")
                decoded_preds.append("")
        
        for label in labels:
            try: 
                label = [l if l != -100 else tokenizer.pad_token_id for l in label]
                 
                label = [l for l in label if 0 <= l < vocab_size]
                decoded_label = tokenizer.decode(label, skip_special_tokens=True)
                decoded_labels.append(decoded_label)
            except Exception as e:
                print(f"Error decoding label: {e}")
                decoded_labels.append("")
      
        bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])["score"]
        chrf_score = chrf.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])["score"]
        
        return {
            "bleu": bleu_score,
            "chrf": chrf_score
        }
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        return {
            "bleu": 0.0,
            "chrf": 0.0
        }
 
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-base-hi-bhili-finetuned",
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=8000,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    #gradient_accumulation_steps=16,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=3,
    fp16=False,
    report_to="tensorboard",
    metric_for_best_model="chrf",
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
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)
 
trainer.train()
 
try:
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:", eval_results)
except Exception as e:
    print(f"Error during evaluation: {e}")
 
model.save_pretrained("./mt5-base-hi-bhili-finetuned")
tokenizer.save_pretrained("./mt5-base-hi-bhili-finetuned")

print("Fine-Tuning Complete! Model and tokenizer saved.")