import pandas as pd
import re
import sys

def split_go_terms_by_branch(input_tsv, output_prefix):
    for i, branch in enumerate(['molecular function', 'biological process', 'cellular component']):
        # Read the input CSV file
        
        branch_name = f'Gene Ontology ({branch})'
        df = pd.read_csv(input_tsv, sep=',', usecols=["Entry", branch_name])

        # Drop rows with missing GO terms
        df = df.dropna(subset=[branch_name])

        # Expand the GO terms so that each row corresponds to one protein-GO term association
        pattern = re.compile(r"\[(GO:\d+)\]")
        exploded_df = df.assign(GO_Term=df[branch_name].str.findall(pattern)).explode('GO_Term')

        branch_fname = '_'.join(branch.split(' '))
        # Write out to separate CSV files
        exploded_df[["Entry", "GO_Term"]].to_csv(f"{output_prefix}_{branch_fname}.csv", index=False)

input_annot_file = sys.argv[1]
output_prefix = sys.argv[2]
split_go_terms_by_branch(input_annot_file, output_prefix)

