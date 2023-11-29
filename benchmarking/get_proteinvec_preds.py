import pandas as pd
import re
import sys

def split_go_terms_by_branch(input_tsv, output_prefix):
    pred_branch_names = ['Predicted_MFO', 'Predicted_BPO', 'Predicted_CCO']
    pred_distance_cols = ['Distance_MFO', 'Distance_BPO', 'Distance_CCO']
    for i, branch in enumerate(['molecular function', 'biological process', 'cellular component']):
        # Read the input CSV file
        
        branch_name = f'Gene Ontology ({branch})'
        pred_branch_name = pred_branch_names[i] # predicted terms
        pred_distance_col = pred_distance_cols[i] # predicted scores
        df = pd.read_csv(input_tsv, sep='\t', usecols=["Entry", branch_name, pred_branch_name, pred_distance_col])

        # Drop rows with missing GO terms
        df = df.dropna(subset=[branch_name])

        # Expand the GO terms so that each row corresponds to one protein-GO term association
        pattern = re.compile(r"\[(GO:\d+)\]")
        
        #exploded_df = df.assign(GO_Term=df[branch_name].str.findall(pattern)).explode(pred_branch_name)
        exploded_df = df.copy()
        exploded_df[pred_branch_name] = df[pred_branch_name].str.findall(pattern)
        exploded_df = exploded_df.explode(pred_branch_name)

        branch_fname = '_'.join(branch.split(' '))
        # Write out to separate CSV files
        new_col_names = {"Entry": 'UniProt_ID', pred_branch_name: 'GO_Term', pred_distance_col: 'Score'}
        exploded_df[["Entry", pred_branch_name, pred_distance_col]].rename(columns=new_col_names).to_csv(f"{output_prefix}_{branch_fname}.csv", index=False)

# Example usage:
input_annot_file = sys.argv[1]
output_prefix = sys.argv[2]
split_go_terms_by_branch(input_annot_file, output_prefix)

