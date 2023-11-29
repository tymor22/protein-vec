import pandas as pd
import sys

# Load the TSV file
tsv_file_path = sys.argv[1]
fasta_output_path = sys.argv[2]

tsv_df = pd.read_csv(tsv_file_path, sep='\t', usecols=['Entry', 'Sequence'])

tsv_df = tsv_df.drop_duplicates(subset='Sequence')


# Write to FASTA format
with open(fasta_output_path, 'w') as fasta_file:
    for idx, row in tsv_df.iterrows():
        sequence = row['Sequence'][:3000]
        fasta_file.write(f">{row['Entry']}\n")
        fasta_file.write(f"{sequence}\n")
