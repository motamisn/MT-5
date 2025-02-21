import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define model and tokenizer path
checkpoint_path = "/home/pooja/Prince/MT-5/MT-5-base/mt5-base-hi-bhili-finetuned/checkpoint-40000"
csv_file = "/home/pooja/Prince/MT-5/MT-5-base/filter_test.csv"
output_csv = "/home/pooja/Prince/MT-5/MT-5-base/filter_test_40000.csv"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load BLEU, chrF, and chrF2 metrics
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

# Load dataset
df = pd.read_csv(csv_file)

def generate_prediction(text):
    """Generate Bhili translation from Hindi text using the model."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_scores(prediction, reference):
    """Compute BLEU, chrF, and chrF2 scores for the given prediction and reference."""
    if not prediction or not reference:
        return 0.0, 0.0, 0.0
    bleu_score = float(bleu.compute(predictions=[prediction], references=[[reference]])["score"])
    chrf_score = float(chrf.compute(predictions=[prediction], references=[[reference]])["score"])
    chrf2_score = float(chrf.compute(predictions=[prediction], references=[[reference]], word_order=2)["score"])
    return bleu_score, chrf_score, chrf2_score

# Initialize new columns
df["Predicted_Bhili"] = ""
df["BLEU"] = 0.0
df["chrF"] = 0.0
df["chrF2"] = 0.0

# Lists to store scores for averaging
bleu_scores, chrf_scores, chrf2_scores = [], [], []

# Iterate through rows and compute predictions and scores
for index, row in df.iterrows():
    hindi_text = str(row["Hindi"]) if pd.notna(row["Hindi"]) else ""
    bhili_reference = str(row["Bhili"]) if pd.notna(row["Bhili"]) else ""
    
    predicted_text = generate_prediction(hindi_text)
    bleu_score, chrf_score, chrf2_score = compute_scores(predicted_text, bhili_reference)
    
    df.at[index, "Predicted_Bhili"] = predicted_text
    df.at[index, "BLEU"] = bleu_score
    df.at[index, "chrF"] = chrf_score
    df.at[index, "chrF2"] = chrf2_score

    # Store scores
    bleu_scores.append(bleu_score)
    chrf_scores.append(chrf_score)
    chrf2_scores.append(chrf2_score)

# Compute average scores
avg_bleu = np.mean(bleu_scores)
avg_chrf = np.mean(chrf_scores)
avg_chrf2 = np.mean(chrf2_scores)

# Print average scores
print("\nFinal Averages:")
print(f"Average BLEU Score: {avg_bleu:.2f}")
print(f"Average chrF Score: {avg_chrf:.2f}")
print(f"Average chrF2 Score: {avg_chrf2:.2f}")

# Save results
df.to_csv(output_csv, index=False)
print(f"\nResults saved to {output_csv}")
