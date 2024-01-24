import argparse
import numpy as np
import pandas as pd
import torch
from utils_search import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    
    # Load the embeddings
    embeddings = np.load(args.target_embeddings)
    # Load the metadata
    lookup_proteins_meta = pd.read_csv(args.target_metadata, sep="\t")
    # Load the query embeddings
    query_embeddings = np.load(args.query_embeddings)

    col_lookup = lookup_proteins_meta[~lookup_proteins_meta[args.aspect].isnull()]
    col_lookup_embeddings = embeddings[col_lookup.index]
    # col_meta_data = col_lookup[args.aspect].values

    lookup_database = load_database(col_lookup_embeddings)

    D, I = query(lookup_database, query_embeddings, args.k)

    #Get metadata for the 1st nearest neighbor
    # near_ids = []
    # for i in range(I.shape[0]):
    #     meta = col_meta_data[I[i]]
    #     near_ids.append(list(meta))       

    # near_ids = np.array(near_ids)

    near_ids_full = []
    for i in range(I.shape[0]):
        # meta = col_lookup.iloc[I[i]].copy()
        meta = col_lookup.iloc[I[i]]
        # meta["distance"] = D[i]
        # meta["query_index"] = [i for _ in range(len(meta))]
        near_ids_full.append(meta)

    df_near_ids_full = pd.DataFrame(columns=near_ids_full[0].columns)

    # Append each Series to the DataFrame
    for series in near_ids_full:
        df_near_ids_full = df_near_ids_full.append(series, ignore_index=True)

    # Save the DataFrame as a CSV file
    df_near_ids_full.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    # TODO: Allow for taking in a list of sequences instead of an embedding
    # parser.add_argument("--input_fasta", type=str, help="input fasta file")
    # parser.add_argument("--input_csv", type=str, help="output csv file", default=None)

    parser = argparse.ArgumentParser(description="Search against embeddings")
    parser.add_argument(
        "--query_embeddings", type=str, help="input embeddings", default="query_embeddings.npy"
    )
    parser.add_argument(
        "--target_embeddings",
        type=str,
        help="lookup embeddings",
        default="protein_vec_embeddings/lookup_embeddings.npy",
    )
    parser.add_argument(
        "--target_metadata",
        type=str,
        help="lookup metadata",
        default="protein_vec_embeddings/lookup_embeddings_meta_data.tsv",
    )
    parser.add_argument(
        "--output_csv", type=str, help="output csv results file", default="results.csv"
    )
    parser.add_argument(
        "--k",
        type=int,
        help="number of nearest neighbors for each protein to return",
        default=1,
    )
    parser.add_argument(
        "--aspect",
        type=str,
        help="aspect of the lookup to search against",
        # give the following options
        #  - 'Gene Ontology (biological process)'
        #  - 'Gene Ontology (molecular function)'
        #  - 'Gene Ontology (cellular component)'
        #  - 'Gene3D'
        #  - 'Pfam'
        #  - 'EC number'
        #  - 'Entry' (search everything)
        default="Pfam",
    )

    args = parser.parse_args()

    main(args)
