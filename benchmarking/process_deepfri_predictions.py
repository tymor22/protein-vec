import json
import sys
import pandas as pd
import numpy as np

# Adjusting the function to handle the dictionary structure of predictions
def process_deepfri_predictions_to_csv_v9(json_file, output_csv):
    """
    Process the DeepFRI predictions in JSON format, aggregate them at the protein level, 
    and save them as a CSV.
    
    Args:
    - json_file (str): Path to the JSON file with DeepFRI predictions.
    - output_csv (str): Path to save the processed predictions as a CSV.
    """
    
    # Load the DeepFRI predictions from the JSON file
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    # Extract protein IDs, GO terms, and scores
    pdb_chains = predictions['pdb_chains']
    goterms = predictions['goterms']
    Y_hat = predictions['Y_hat']
    
    # Extract predictions for each protein-GO term pair
    protein_predictions = {}
    for protein_or_chain, scores in zip(pdb_chains, Y_hat):
        # Extracting protein ID (without chain specification)
        protein_id = protein_or_chain.split('_chain_')[0]
        
        # Storing scores for the protein
        if protein_id not in protein_predictions:
            protein_predictions[protein_id] = []
        protein_predictions[protein_id].append(scores)
    
    # Averaging the scores for each protein
    averaged_predictions = []
    for protein_id, protein_scores in protein_predictions.items():
        avg_scores = np.mean(protein_scores, axis=0)
        for go_term, score in zip(goterms, avg_scores):
            averaged_predictions.append([protein_id, go_term, score])
    
    # Convert the list of predictions to a DataFrame
    df_predictions = pd.DataFrame(averaged_predictions, columns=['UniProt_ID', 'GO_Term', 'Score'])
    
    # Save the DataFrame to a CSV file
    df_predictions.to_csv(output_csv, index=False)

# Process the DeepFRI predictions with the updated function
input_path = sys.argv[1]
output_csv_path = sys.argv[2]
process_deepfri_predictions_to_csv_v9(input_path, output_csv_path)

