import csv
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import sys
import matplotlib.pyplot as plt

def generate_naive_predictions(annotation_df, proteins_to_predict):
    # Calculate the frequency of each GO term
    go_term_freq = annotation_df['GO_Term'].value_counts() / len(annotation_df)

    # Generate predictions for each protein
    predictions = []
    for protein in proteins_to_predict:
        for go_term, freq in go_term_freq.items():
            predictions.append([protein, go_term, freq])

    return pd.DataFrame(predictions, columns=['UniProt_ID', 'GO_Term', 'Score'])

def plot_curve(x, y, title="", x_label="", y_label="", color="blue"):
    """
    Plots a curve given x and y coordinates.
    
    Parameters:
    - x: List or array of x coordinates
    - y: List or array of y coordinates
    - title (optional): Title of the plot
    - x_label (optional): Label for the x-axis
    - y_label (optional): Label for the y-axis
    - color (optional): Color of the curve. Default is blue.
    """
    
    plt.figure(figsize=(10,6))
    plt.plot(x, y, color=color, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def micro_AUPR(label, score):
    """Computing AUPR (micro-averaging)"""
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)  # python

    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision

    pr = np.trapz(y, x)

    return pr

def get_ground_truth(filename):
    """
    Load ground truth labels from the specified file.
    """
    ground_truth = set()
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            protein, go_term = row
            ground_truth.add((protein, go_term))
    return ground_truth

def compute_aupr(predictions_file, ground_truth_file):
    # Load data
    pred_df = pd.read_csv(predictions_file)
    gt_df = pd.read_csv(ground_truth_file, names=['UniProt_ID', 'GO_Term'])
    
    # Extract proteins present in the ground truth
    ground_truth_proteins = set(gt_df['UniProt_ID'].unique())
    
    # Filter predictions based on proteins present in ground truth
    pred_df = pred_df[pred_df['UniProt_ID'].isin(ground_truth_proteins)]
    
    # Extract all unique GO terms
    all_go_terms = set(pred_df['GO_Term'].unique()).union(set(gt_df['GO_Term'].unique()))
    
    # Initialize y_true and y_score matrices
    y_true = np.zeros((len(ground_truth_proteins), len(all_go_terms)))
    y_score = np.zeros((len(ground_truth_proteins), len(all_go_terms)))
    
    protein_to_idx = {protein: idx for idx, protein in enumerate(ground_truth_proteins)}
    go_to_idx = {go: idx for idx, go in enumerate(all_go_terms)}
    
    # Populate y_true and y_score
    for _, row in gt_df.iterrows():
        protein_idx = protein_to_idx[row['UniProt_ID']]
        go_idx = go_to_idx[row['GO_Term']]
        y_true[protein_idx, go_idx] = 1
    
    for _, row in pred_df.iterrows():
        protein_idx = protein_to_idx[row['UniProt_ID']]
        go_idx = go_to_idx[row['GO_Term']]
        y_score[protein_idx, go_idx] = row['Score']
    
    # Compute macro and micro AUPRs
    macro_aupr = average_precision_score(y_true, y_score, average='macro')
    micro_aupr = average_precision_score(y_true, y_score, average='micro')
    
    return macro_aupr, micro_aupr


def dataframes_to_matrices(gt_df, pred_df, proteins, go_terms):
    """
    Convert ground truth and prediction dataframes to dense matrices.
    
    Parameters:
    - gt_df: Ground truth dataframe
    - pred_df: Predictions dataframe
    - proteins: List of unique proteins
    - go_terms: List of unique GO terms
    
    Returns:
    - y_true: Dense matrix representation of ground truth
    - y_score: Dense matrix representation of predictions
    """
    # Map protein IDs and GO term IDs to integer indices
    protein_to_idx = {protein: idx for idx, protein in enumerate(proteins)}
    go_term_to_idx = {go_term: idx for idx, go_term in enumerate(go_terms)}
    
    # Convert protein IDs and GO term IDs in the ground truth dataframe to integer indices
    row_indices_gt = gt_df['UniProt_ID'].map(protein_to_idx).values
    col_indices_gt = gt_df['GO_Term'].map(go_term_to_idx).values
    
    # Create a sparse COO matrix for the ground truth using the mapped indices
    # All entries are 1 since we are creating a binary matrix
    sparse_matrix_gt = coo_matrix((np.ones(len(row_indices_gt)), (row_indices_gt, col_indices_gt)), 
                                  shape=(len(proteins), len(go_terms)))
    
    # Convert the sparse matrix to a dense numpy array
    y_true = sparse_matrix_gt.toarray()
    
    # Convert protein IDs and GO term IDs in the predictions dataframe to integer indices
    row_indices_pred = pred_df['UniProt_ID'].map(protein_to_idx).values
    col_indices_pred = pred_df['GO_Term'].map(go_term_to_idx).values
    
    # Use the scores from the predictions dataframe as data for the sparse COO matrix
    data_pred = pred_df['Score'].values
    
    # Create a sparse COO matrix for the predictions using the mapped indices and scores
    sparse_matrix_pred = coo_matrix((data_pred, (row_indices_pred, col_indices_pred)), 
                                   shape=(len(proteins), len(go_terms)))
    
    # Convert the sparse matrix to a dense numpy array
    y_score = sparse_matrix_pred.toarray()
    
    return y_true, y_score


def compute_auprs_with_missing_proteins(predictions_file, ground_truth_file, prot_list=None):
    # Read files
    pred_df = pd.read_csv(predictions_file)
    gt_df = pd.read_csv(ground_truth_file, names=['UniProt_ID', 'GO_Term'])
    if prot_list is not None:
        print('subsetting ground truth dataframe...num prots in list: ' + str(len(prot_list)))
        gt_df = gt_df[gt_df['UniProt_ID'].isin(prot_list)]

    # Extract unique proteins and GO terms from the ground truth
    unique_proteins_gt = set(gt_df['UniProt_ID'].unique())
    print('Number of ground truth prots: ' + str(len(unique_proteins_gt)))
    unique_go_terms_gt = set(gt_df['GO_Term'].unique())

    # Extract unique proteins from the predictions
    unique_proteins_pred = set(pred_df['UniProt_ID'].unique())

    # Find proteins that are in ground truth but missing from predictions
    missing_proteins = unique_proteins_gt - unique_proteins_pred
    print('Prediction file missing prots: ' + str(len(missing_proteins)))
    #import ipdb; ipdb.set_trace()

    pred_df = pred_df[pred_df['UniProt_ID'].isin(gt_df['UniProt_ID'])] # remove extraneous predictions for proteins not in ground truth
    pred_df = pred_df[pred_df['GO_Term'].isin(unique_go_terms_gt)] # remove extra GO terms

    # now pred_df's proteins and go terms are subsets of gt_df
    assert set(pred_df['UniProt_ID'].unique()).issubset(set(gt_df['UniProt_ID'].unique()))
    assert set(pred_df['GO_Term'].unique()).issubset(set(gt_df['GO_Term'].unique()))

    # Create matrix representation
    y_true, y_score = dataframes_to_matrices(gt_df, pred_df, sorted(list(unique_proteins_gt)), sorted(list(unique_go_terms_gt)))

    # Compute macro AUPR
    #average_precision_macro = average_precision_score(y_true, y_score, average="macro")
    #go_term_auprs = [micro_AUPR(y_true[:,i],y_score[:,i]) for i in range(0, len(unique_go_terms_gt))]
    go_term_pr_curves = [precision_recall_curve(y_true[:,i], y_score[:,i]) for i in range(0, len(unique_go_terms_gt))]
    go_auprs = [auc(go_term_pr_curve[1], go_term_pr_curve[0]) for go_term_pr_curve in go_term_pr_curves]
    #[plot_curve(go_term_pr_curve[1], go_term_pr_curve[0]) for go_term_pr_curve in go_term_pr_curves]
    #average_precision_macro = sum(go_term_auprs)/len(go_term_auprs)
    aupr_macro = sum(go_auprs)/len(go_auprs)

    # Compute micro AUPR
    #average_precision_micro = average_precision_score(y_true, y_score, average="micro")
    #average_precision_micro = micro_AUPR(y_true, y_score)
    pr_curve = precision_recall_curve(y_true.flatten(), y_score.flatten())
    #plot_curve(pr_curve[1], pr_curve[0])
    aupr_micro = auc(pr_curve[1], pr_curve[0])
    

    return aupr_macro, aupr_micro, len(missing_proteins), list(unique_proteins_pred)

'''
def generate_single_latex_file(data_dict, output_file):
    with open(output_file, 'w') as f:
        for subset_name, subset_data in data_dict.items():
            for branch, branch_data in subset_data.items():
                # Determine unique methods and metrics
                methods = list(branch_data.keys())
                metrics = list(branch_data[methods[0]].keys())
                
                # Construct header
                header = "Branch & "
                for method in methods:
                    for metric in metrics:
                        method_escaped = method.replace('_', r'\_')
                        header += f"{method_escaped} ({metric}) & "
                header = header.rstrip(" & ")
                header += " \\\\ \\hline\n"
                
                # Construct row for the current branch
                branch_escaped = branch.replace('_', r'\_')
                row = f"{branch_escaped} & "
                for method in methods:
                    for metric in metrics:
                        row += f"{branch_data[method][metric]:.4f} & "
                row = row.rstrip(" & ")
                row += " \\\\ \\hline"
                
                # Compile table
                subset_name_escaped = subset_name.replace('_', r'\_')
                latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|" + "c|" * (len(methods) * len(metrics)) + "}\n\\hline\n"
                latex_table += header
                latex_table += row
                latex_table += f"\n\\end{{tabular}}\n\\caption{{Results for {subset_name_escaped} - {branch_escaped}}}\n\\end{{table}}\n\n"
                
                # Write to file
                f.write(latex_table)
'''

def generate_single_latex_file(data_dict, output_file):
    main_method = 'Protein-Vec'
    with open(output_file, 'w') as f:
        for subset_name, subset_data in data_dict.items():
            for branch, branch_data in subset_data.items():
                # Determine unique methods and metrics
                methods = list(branch_data.keys())
                method_headers = [method.replace('_', r'\_') for method in methods]
                '''
                method_headers = []
                for method in methods:
                    if method == main_method:
                        method_headers.append("\\textbf{" + method.replace('_', r'\_') + "}")
                    else:
                        method_headers.append(method.replace('_', r'\_'))
                '''
                metrics = list(branch_data[methods[0]].keys())
                
                # Construct header
                header = "Metric & " + " & ".join(method_headers) + " \\\\ \\hline\n"
                
                # Compile table rows
                rows = []
                for metric in metrics:
                    row = f"{metric} & "
                    for method in methods:
                        if method == main_method:
                            row += "\\textbf{" + f"{branch_data[method][metric]:.4f}" + "} & "
                        else:
                            row += f"{branch_data[method][metric]:.4f} & "
                    row = row.rstrip(" & ")
                    rows.append(row + " \\\\ \\hline")
                
                # Compile table
                subset_name_escaped = subset_name.replace('_', r'\_')
                branch_escaped = branch.replace('_', r'\_')
                latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|" + "c|" * len(methods) + "}\n\\hline\n"
                latex_table += header
                latex_table += "\n".join(rows)
                latex_table += f"\n\\end{{tabular}}\n\\caption{{Results for {subset_name_escaped} - {branch_escaped}}}\n\\end{{table}}\n\n"
                
                # Write to file
                f.write(latex_table)


prot_list = None
if len(sys.argv) > 1:
    prot_list_file = sys.argv[1]
    print('Using ' + prot_list_file + ' to restrict performances only over these proteins.')
    prot_list = list(open(prot_list_file, 'r').readlines())

# Compute AUPR for each branch
branches = ["Molecular_Function", "Cellular_Component", "Biological_Process"]
acronyms = {'Molecular_Function': 'mf', 'Cellular_Component': 'cc', 'Biological_Process': 'bp'}
metrics = {'test_set_perfs': {}, 'deepFRI_subset_perfs':{}}
for branch in branches:
    deepGOPlus_predictions_file = f"test_proteins_{branch}_Predictions_from_deepGOPlus.csv"
    deepFRI_predictions_file = f"deepfri_single_chain_{branch}_preds.csv"
    #ground_truth_file = f"{branch}_new.csv"
    ground_truth_file = f"fixed_annotation_files_{branch.lower()}.csv"
    #prott5_predictions_file = f"prott5_test_preds_{branch}.csv"
    #prott5_predictions_file = f"prott5_xl_u50_test_preds_{branch}.csv"
    prott5_predictions_file = f"prott5_xl_u50_euc_dist_test_preds_{branch}.csv"
    acronym = acronyms[branch]
    #proteinvec_predictions_file = f'proteinvec_pred_{acronym}.csv'
    proteinvec_predictions_file = f'proteinvec_pred_fixed_{branch.lower()}.csv'

    training_ground_truth_file = f"training_data_{branch}.csv"
    
    #macro_aupr, micro_aupr = compute_aupr(predictions_file, ground_truth_file)
    print('Protein-Vec')
    proteinvec_macro_aupr, proteinvec_micro_aupr, proteinvec_missing_proteins, proteinvec_prots = compute_auprs_with_missing_proteins(proteinvec_predictions_file, ground_truth_file, prot_list=prot_list)
    #print('Doing it again only over those protein-vec prots')
    #prot_list = proteinvec_prots
    proteinvec_macro_aupr, proteinvec_micro_aupr, proteinvec_missing_proteins, proteinvec_prots = compute_auprs_with_missing_proteins(proteinvec_predictions_file, ground_truth_file, prot_list=prot_list)
    print('deepGOPlus')
    deepGOPlus_macro_aupr, deepGOPlus_micro_aupr, deepGOPlus_missing_proteins, deepGOPlus_prots = compute_auprs_with_missing_proteins(deepGOPlus_predictions_file, ground_truth_file, prot_list=prot_list)
    print('deepFRI')
    deepFRI_macro_aupr, deepFRI_micro_aupr, deepFRI_missing_proteins, deepFRI_prots = compute_auprs_with_missing_proteins(deepFRI_predictions_file, ground_truth_file, prot_list=prot_list)
    print('prott5')
    prott5_macro_aupr, prott5_micro_aupr, prott5_missing_proteins, prott5_prots = compute_auprs_with_missing_proteins(prott5_predictions_file, ground_truth_file, prot_list=prot_list)
    print(f'~~~~~~~~~Performances for all test proteins for {branch}:~~~~~~~~~')
    print(f"deepGOPlus AUPR for {branch}: {deepGOPlus_macro_aupr:.4f} {deepGOPlus_micro_aupr:.4f}")
    print(f"DeepFRI AUPR for {branch}: {deepFRI_macro_aupr:.4f} {deepFRI_micro_aupr:.4f}")
    print(f"prott5 AUPR for {branch}: {prott5_macro_aupr:.4f} {prott5_micro_aupr:.4f}")
    print(f"Protein-Vec AUPR for {branch}: {proteinvec_macro_aupr:.4f} {proteinvec_micro_aupr:.4f}")

    
    # Assuming you have loaded your ground truth annotations into a DataFrame called gt_df
    test_gt_df = pd.read_csv(ground_truth_file, names=['UniProt_ID', 'GO_Term'])
    training_gt_df = pd.read_csv(training_ground_truth_file, names=['UniProt_ID', 'GO_Term'])
    proteins_to_predict = list(test_gt_df['UniProt_ID'].unique())
    naive_predictions_df = generate_naive_predictions(training_gt_df, proteins_to_predict) # base frequencies off of training_gt_df, predict for proteins that are coming from test_gt_df

    # Save the naive predictions to a CSV file if you want
    naive_predictions_df.to_csv("naive_predictions_" + str(branch) + ".csv", index=False)
    naive_macro_aupr, naive_micro_aupr, naive_missing_proteins, naive_prots = compute_auprs_with_missing_proteins("naive_predictions_" + str(branch) + ".csv", ground_truth_file, prot_list=prot_list)
    print(f"Naive AUPR for {branch}: {naive_macro_aupr:.4f} {naive_micro_aupr:.4f}")

    metrics['test_set_perfs'][branch] = {
            'Protein-Vec': {'Macro AUPR': proteinvec_macro_aupr, 'Micro AUPR': proteinvec_micro_aupr},
            "deepGOPlus": {'Macro AUPR': deepGOPlus_macro_aupr, 'Micro AUPR': deepGOPlus_micro_aupr},
            'prott5': {'Macro AUPR': prott5_macro_aupr, 'Micro AUPR': prott5_micro_aupr},
    'Naive': {'Macro AUPR': naive_macro_aupr, 'Micro AUPR': naive_micro_aupr}}

    # To compare only with deepFRI proteins
    print('deepGOPlus')
    deepGOPlus_macro_aupr, deepGOPlus_micro_aupr, deepGOPlus_missing_proteins, deepGOPlus_prots = compute_auprs_with_missing_proteins(deepGOPlus_predictions_file, ground_truth_file, prot_list=deepFRI_prots)
    print('deepFRI')
    deepFRI_macro_aupr, deepFRI_micro_aupr, deepFRI_missing_proteins, deepFRI_prots = compute_auprs_with_missing_proteins(deepFRI_predictions_file, ground_truth_file, prot_list=deepFRI_prots)
    print('prott5')
    prott5_macro_aupr, prott5_micro_aupr, prott5_missing_proteins, prott5_prots = compute_auprs_with_missing_proteins(prott5_predictions_file, ground_truth_file, prot_list=deepFRI_prots)
    print('Protein-Vec')
    proteinvec_macro_aupr, proteinvec_micro_aupr, proteinvec_missing_proteins, proteinvec_prots = compute_auprs_with_missing_proteins(proteinvec_predictions_file, ground_truth_file, prot_list=deepFRI_prots)
    naive_predictions_df = generate_naive_predictions(training_gt_df, deepFRI_prots)
    naive_predictions_df.to_csv("naive_predictions_structure_prots_only_" + str(branch) + ".csv", index=False)
    naive_macro_aupr, naive_micro_aupr, naive_missing_proteins, naive_prots = compute_auprs_with_missing_proteins("naive_predictions_structure_prots_only_" + str(branch) + ".csv", ground_truth_file, prot_list=prot_list)
    print(f'~~~~~~~~~Performances for DeepFRI proteins ONLY for {branch}:~~~~~~~~~')
    print(f"deepGOPlus AUPR for {branch}: {deepGOPlus_macro_aupr:.4f} {deepGOPlus_micro_aupr:.4f}")
    print(f"DeepFRI AUPR for {branch}: {deepFRI_macro_aupr:.4f} {deepFRI_micro_aupr:.4f}")
    print(f"prott5 AUPR for {branch}: {prott5_macro_aupr:.4f} {prott5_micro_aupr:.4f}")
    print(f"Protein-Vec AUPR for {branch}: {proteinvec_macro_aupr:.4f} {proteinvec_micro_aupr:.4f}")
    print(f"Naive AUPR for {branch}: {naive_macro_aupr:.4f} {naive_micro_aupr:.4f}")
    metrics['deepFRI_subset_perfs'][branch] = {
    'Protein-Vec': {'Macro AUPR': proteinvec_macro_aupr, 'Micro AUPR': proteinvec_micro_aupr},
    'deepGOPlus': {'Macro AUPR': deepGOPlus_macro_aupr, 'Micro AUPR': deepGOPlus_micro_aupr},
    'DeepFRI': {'Macro AUPR': deepFRI_macro_aupr, 'Micro AUPR': deepFRI_micro_aupr},
    'prott5': {'Macro AUPR': prott5_macro_aupr, 'Micro AUPR': prott5_micro_aupr},
    'Naive': {'Macro AUPR': naive_macro_aupr, 'Micro AUPR': naive_micro_aupr}
    }

generate_single_latex_file(metrics, './tables.tex')
#predictions_file = sys.argv[1]
#ground_truth_file = sys.argv[2]

#macro_aupr, micro_aupr = compute_aupr(predictions_file, ground_truth_file)
#macro_aupr, micro_aupr, missing_proteins = compute_auprs_with_missing_proteins(predictions_file, ground_truth_file)
#print(f"AUPR: {macro_aupr:.4f} {micro_aupr:.4f}")

