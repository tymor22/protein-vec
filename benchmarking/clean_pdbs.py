import os
import sys
import argparse

def clean_pdb_files(input_directory, output_directory, min_residue_threshold=10, single_chain_only=False):
    """
    Cleans the PDB files located in the input directory and saves the cleaned files to the output directory.
    
    Args:
    - input_directory (str): Path to the directory containing the original PDB files.
    - output_directory (str): Path to the directory where cleaned PDB files should be saved.
    - min_residue_threshold (int): Minimum number of residues a chain should have after cleaning.
    """
    import os
    
    # List of standard amino acid residues
    standard_amino_acids = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL"
    ]
    
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def remove_hetatms_and_non_standard_residues(pdb_lines):
        """Removes HETATM entries and non-standard amino acid residues from PDB lines."""
        cleaned_lines = [line for line in pdb_lines if not line.startswith("HETATM") and line[17:20].strip() in standard_amino_acids]
        return cleaned_lines
    
    def split_chains(pdb_lines):
        chains = {}
        for line in pdb_lines:
            if line.startswith("ATOM"):
                chain_id = line[21]
                if chain_id not in chains:
                    chains[chain_id] = []
                chains[chain_id].append(line)
        return chains
    
    def fix_residue_numbering(pdb_lines):
        new_lines = []
        prev_chain_id = None
        residue_counter = 0
        prev_residue_num = None
        for line in pdb_lines:
            if line.startswith("ATOM"):
                chain_id = line[21]
                residue_num = int(line[22:26])

                if prev_chain_id != chain_id:
                    residue_counter = 1
                    prev_residue_num = residue_num

                if prev_residue_num != residue_num:
                    residue_counter += 1

                adjusted_line = line[:22] + f"{residue_counter:>4}" + line[26:]
                new_lines.append(adjusted_line)

                prev_chain_id = chain_id
                prev_residue_num = residue_num
            else:
                new_lines.append(line)
        return new_lines

    def ensure_CA_for_all_residues(chain_lines):
        """Ensures that every residue in the chain has a C-alpha atom."""
        residues_with_CA = set(line[22:26] for line in chain_lines if line.startswith("ATOM") and line[12:16].strip() == "CA")
        return [line for line in chain_lines if line[22:26] in residues_with_CA]
    
    # Process each PDB file in the input directory
    single_chain_prots = 0
    for filename in os.listdir(input_directory):
        if filename.endswith(".pdb"):
            filepath = os.path.join(input_directory, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Clean the PDB
            lines_cleaned = remove_hetatms_and_non_standard_residues(lines)
            chains = split_chains(lines_cleaned)

            if single_chain_only and len(chains) > 1: 
                single_chain_prots += 1
                continue
            
            # Save each chain separately with fixed residue numbering, only if it meets the residue threshold
            for chain_id, chain_lines in chains.items():
                chain_lines_with_CA = ensure_CA_for_all_residues(chain_lines)
                if len(set([line[22:26] for line in chain_lines_with_CA if line.startswith("ATOM")])) >= min_residue_threshold:
                    fixed_lines = fix_residue_numbering(chain_lines_with_CA)
                    output_filepath = os.path.join(output_directory, f"{filename[:-4]}_chain_{chain_id}_fixed.pdb")
                    with open(output_filepath, 'w') as f:
                        f.writelines(fixed_lines)
    if single_chain_only: 
        print('Number of single chain prots: ' + str(single_chain_prots))

# The updated function ensures that every residue in the cleaned PDB files has a C-alpha atom.


if __name__ == '__main__':
    # Example usage
    # clean_pdb_files("path/to/input/directory", "path/to/output/directory")
    parser = argparse.ArgumentParser(description='Clean PDB files by removing HETATMs and renumbering residues.')
    parser.add_argument('input_dir', type=str, help='Path to directory containing input PDB files.')
    parser.add_argument('output_dir', type=str, help='Path to directory to save cleaned PDB files.')
    parser.add_argument('--single_chain_only', action='store_true', help='Process only PDBs with a single chain.')
    args = parser.parse_args()

    clean_pdb_files(args.input_dir, args.output_dir, min_residue_threshold=50, single_chain_only=args.single_chain_only)
