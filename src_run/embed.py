import argparse
import numpy as np
import pandas as pd
import torch
from model_protein_moe import trans_basic_block, trans_basic_block_Config
from utils_search import *
from transformers import T5EncoderModel, T5Tokenizer
import gc
import numpy as np
import pandas as pd
from collections import defaultdict
import faiss
from Bio import SeqIO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fasta_to_csv(input_fasta, output_csv):
    # Read the .fasta file
    records = SeqIO.parse(input_fasta, "fasta")

    # Create a list to store the data
    data = []

    # Extract the Accession and Sequence from each record
    for record in records:
        accession = record.id
        sequence = str(record.seq)
        data.append([accession, sequence])

    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=["Accession", "Sequence"])

    # Save the DataFrame as a CSV file
    if output_csv is not None:
        df.to_csv(output_csv, index=False)

    return df


def load_vec_model(vec_model_cpnt, vec_model_config):
    # Load the model
    vec_model_config = trans_basic_block_Config.from_json(vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(
        vec_model_cpnt, config=vec_model_config
    )
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()
    return model_deep


def get_tokenizer(ProtTrans_model, ProtTrans_tokenizer):
    # Load the ProtTrans model and ProtTrans tokenizer
    tokenizer = T5Tokenizer.from_pretrained(ProtTrans_tokenizer, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtTrans_model)
    gc.collect()

    model = model.to(device)
    model = model.eval()
    return model, tokenizer


def get_masks():
    # This is a forward pass of the Protein-Vec model
    # Every aspect is turned on (therefore no masks)
    sampled_keys = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    all_cols = np.array(["TM", "PFAM", "GENE3D", "ENZYME", "MFO", "BPO", "CCO"])
    masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
    masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None, :]
    return masks


def load_data(input_csv, input_fasta, output_csv):
    # Load the data
    if input_csv is not None:
        df = pd.read_csv(input_csv)

    # If the data is not already in a CSV file, convert the .fasta file to a CSV file
    if input_fasta is not None:
        df = fasta_to_csv(input_fasta, output_csv)

    return df


def main(args):
    if args.input_csv is None and args.input_fasta is None:
        raise ValueError("Must provide either input_csv or input_fasta")

    df = load_data(args.input_csv, args.input_fasta, args.output_csv)

    model, tokenizer = get_tokenizer(args.ProtTrans_model, args.ProtTrans_tokenizer)
    model_deep = load_vec_model(args.vec_model_cpnt, args.vec_model_config)

    masks = get_masks()

    embeddings = encode(
        df["Sequence"].values, model_deep, model, tokenizer, masks, device, args.batch_size
    )
    # save the embeddings
    np.save(args.output_file, embeddings)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Embedding generation")
    parser.add_argument("--input_fasta", type=str, help="input fasta file")
    parser.add_argument("--input_csv", type=str, help="output csv file", default=None)
    parser.add_argument("--output_csv", type=str, help="output csv file", default=None)
    parser.add_argument(
        "--vec_model_cpnt",
        type=str,
        help="vec model checkpoint",
        default="protein_vec_models/protein_vec.ckpt",
    )
    parser.add_argument(
        "--vec_model_config",
        type=str,
        help="vec model config",
        default="protein_vec_models/protein_vec_params.json",
    )
    parser.add_argument(
        "--ProtTrans_model",
        type=str,
        help="ProtTrans model",
        default="Rostlab/prot_t5_xl_uniref50",
    )
    parser.add_argument(
        "--ProtTrans_tokenizer",
        type=str,
        help="ProtTrans tokenizer",
        default="Rostlab/prot_t5_xl_uniref50",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="output file for embeddings",
        default="embeddings.npy",
    )
    parser.add_argument(
        "--batch_size", type=int, help="batch size for embeddings", default=8
    )

    args = parser.parse_args()

    main(args)
