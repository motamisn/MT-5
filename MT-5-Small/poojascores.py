import pandas as pd
import evaluate

# Load the ChrF metric
chrf = evaluate.load("chrf")

# Load the CSV file
file_path = '/content/mankibaat5e__33k.csv'  # Replace with actual path if needed
data = pd.read_csv(file_path)

# Ensure the CSV has the correct columns
assert 'Hindi' in data.columns, "Column 'Hindi' not found in CSV."
assert 'Original_bhili' in data.columns, "Column 'Original_Bhili' not found in CSV."
assert 'Predicted_Bhili' in data.columns, "Column 'Predicted_Bhili' not found in CSV."

# Extract the references and predictions
references = data['Original_bhili'].apply(lambda x: [x]).tolist()
predictions = data['Predicted_Bhili'].tolist()

# Compute ChrF and ChrF2 scores for each row and store them
chrf_scores = []
chrf2_scores = []

for pred, ref in zip(predictions, references):
    chrf_result = chrf.compute(predictions=[pred], references=[ref])
    chrf2_result = chrf.compute(predictions=[pred], references=[ref], word_order=2)
    chrf_scores.append(chrf_result['score'])
    chrf2_scores.append(chrf2_result['score'])

# Add the scores to the DataFrame
data['ChrF_Score'] = chrf_scores
data['ChrF2_Score'] = chrf2_scores

# Calculate average scores
average_chrf = sum(chrf_scores) / len(chrf_scores)
average_chrf2 = sum(chrf2_scores) / len(chrf2_scores)

# Save the updated DataFrame to a new CSV file
output_file_path = 'output_mkbjan5e_33k.csv'
data.to_csv(output_file_path, index=False)

(output_file_path, average_chrf, average_chrf2)
