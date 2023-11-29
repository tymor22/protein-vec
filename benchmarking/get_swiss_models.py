import requests
import sys
import os

def fetch_pdb_from_swissmodel(uniprot_id, output_dir="./similar_pdbs"):
    """Fetch PDB file for a given UniProtKB accession number from SWISS-MODEL Repository API.
    
    Args:
    - uniprot_id (str): UniProtKB accession number.
    - output_dir (str): Directory to save the fetched PDB files.
    
    Returns:
    - str: Path to the saved PDB file or None if not found.
    """
    
    base_url = "https://swissmodel.expasy.org/repository/uniprot/"
    params = {
        "sort": "seqsim",  # Sort by sequence similarity
        "provider": "pdb"
    }
    response = requests.get(f"{base_url}{uniprot_id}.pdb?sort=seqsim&provider=pdb", params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        print(f"Found {uniprot_id}")
        pdb_file_path = os.path.join(output_dir, f"{uniprot_id}.pdb")
        with open(pdb_file_path, 'wb') as f:
            f.write(response.content)
        return pdb_file_path
    else:
        print(f"No PDB found for UniProt ID: {uniprot_id}")
        return None

if __name__ == "__main__":
    # Read UniProtKB accession numbers from the file
    with open(sys.argv[1], "r") as f:
        uniprot_ids = [line.strip() for line in f.readlines()]

    for uid in uniprot_ids:
        fetch_pdb_from_swissmodel(uid)

