import pandas as pd
import evaluate
import sacrebleu

# Load the ChrF metric
chrf = evaluate.load("chrf")

# Load the CSV file
file_path = '/home/pooja/Prince/MT-5/MT-5-base/mt5_base_hpc/testset_40000.csv'
data = pd.read_csv(file_path)

# Ensure the CSV has the correct columns
assert 'Hindi' in data.columns, "Column 'Hindi' not found in CSV."
assert 'Bhili' in data.columns, "Column 'Bhili' not found in CSV."
assert 'Predicted_Bhili' in data.columns, "Column 'Predicted_Bhili' not found in CSV."

# Clean and filter data
data = data.dropna(subset=['Bhili', 'Predicted_Bhili'])
data['Bhili'] = data['Bhili'].astype(str)
data['Predicted_Bhili'] = data['Predicted_Bhili'].astype(str)

# Extract the references and predictions
references = data['Bhili'].apply(lambda x: [x]).tolist()
predictions = data['Predicted_Bhili'].tolist()

# Compute ChrF and ChrF2 scores
chrf_scores = []
chrf2_scores = []
spbleu_scores = []

for pred, ref_list in zip(predictions, references):
    if not pred.strip() or not ref_list or not all(ref.strip() for ref in ref_list):
        continue

    chrf_result = chrf.compute(predictions=[pred], references=ref_list)
    chrf2_result = chrf.compute(predictions=[pred], references=ref_list, word_order=2)
    spbleu_result = sacrebleu.sentence_bleu(pred, ref_list)

    chrf_scores.append(chrf_result['score'])
    chrf2_scores.append(chrf2_result['score'])
    spbleu_scores.append(spbleu_result.score)

# Add the scores to the DataFrame
data['ChrF_Score'] = chrf_scores
data['ChrF2_Score'] = chrf2_scores
data['spBLEU_Score'] = spbleu_scores

# Save the updated DataFrame to a new CSV file
output_file_path = '/home/pooja/Prince/MT-5/MT-5-base/mt5_base_hpc/testset_40000scores.csv'
data.to_csv(output_file_path, index=False)

print(f"Saved updated results to {output_file_path}")
print(f"Average ChrF: {sum(chrf_scores) / len(chrf_scores):.2f}, Average ChrF2: {sum(chrf2_scores) / len(chrf2_scores):.2f}, Average spBLEU: {sum(spbleu_scores) / len(spbleu_scores):.2f}")
