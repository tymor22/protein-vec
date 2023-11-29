import sys
import pandas as pd

# Extract and format data for each Gene Ontology branch

def extract_go_terms(entry_column, go_column):
    go_terms = []
    for entry, terms in zip(entry_column, go_column):
        if pd.notna(terms):
            for term in terms.split(';'):
                go_id = term.split('[')[-1].split(']')[0]
                if go_id.startswith('GO'):
                    go_terms.append([entry, go_id])
    return go_terms

# Extract terms for each branch
annot_file = sys.argv[1]

# Load the provided CSV file
data = pd.read_csv(annot_file)

mf_terms = extract_go_terms(data['Entry'], data['Gene Ontology (molecular function)'])
cc_terms = extract_go_terms(data['Entry'], data['Gene Ontology (cellular component)'])
bp_terms = extract_go_terms(data['Entry'], data['Gene Ontology (biological process)'])

# Save to separate files
output_keyword = sys.argv[2]
mf_file_path_new = f'./Molecular_Function_{output_keyword}.csv'
cc_file_path_new = f'./Cellular_Component_{output_keyword}.csv'
bp_file_path_new = f'./Biological_Process_{output_keyword}.csv'

pd.DataFrame(mf_terms, columns=['UniProt_ID', 'GO_Term']).to_csv(mf_file_path_new, index=False, header=False)
pd.DataFrame(cc_terms, columns=['UniProt_ID', 'GO_Term']).to_csv(cc_file_path_new, index=False, header=False)
pd.DataFrame(bp_terms, columns=['UniProt_ID', 'GO_Term']).to_csv(bp_file_path_new, index=False, header=False)
