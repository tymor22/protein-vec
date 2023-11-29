import time
import requests

API_URL = "https://deepgo.cbrc.kaust.edu.sa/deepgo/api/create"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}
API_VERSION = "1.0.14"

def predict_function_with_deepgoplus(fasta_string):
    data = {
        "version": API_VERSION,
        "data_format": "fasta",
        "data": fasta_string
    }
    response = requests.post(API_URL, json=data, headers=HEADERS)
    return response

def handle_deepgoplus_response(response, output_prefix):
    if response.status_code in [200, 201]:
        predictions = response.json().get("predictions", [])
        process_predictions_to_files(predictions, output_prefix)
        return True
    else:
        print(f"Failed to fetch predictions. Server responded with status code {response.status_code}.")
        print(response.headers)
        print(response.text)
        return False

def process_predictions_to_files(predictions, output_prefix):
    # Separate the data for each branch
    molecular_function_data = []
    cellular_component_data = []
    biological_process_data = []

    for prediction in predictions:
        uniprot_id = prediction['protein_info'].split()[0]
        
        for function_category in prediction['functions']:
            category_name = function_category['name']
            for go_data in function_category['functions']:
                go_term, _, score = go_data
                if category_name == "Molecular Function":
                    molecular_function_data.append((uniprot_id, go_term, score))
                elif category_name == "Cellular Component":
                    cellular_component_data.append((uniprot_id, go_term, score))
                elif category_name == "Biological Process":
                    biological_process_data.append((uniprot_id, go_term, score))

    # Save the data to separate files
    with open(f"{output_prefix}_Molecular_Function_Predictions.csv", "a") as mf_file:
        mf_file.write("UniProt_ID,GO_Term,Score\n")
        for row in molecular_function_data:
            mf_file.write(f"{row[0]},{row[1]},{row[2]}\n")

    with open(f"{output_prefix}_Cellular_Component_Predictions.csv", "a") as cc_file:
        cc_file.write("UniProt_ID,GO_Term,Score\n")
        for row in cellular_component_data:
            cc_file.write(f"{row[0]},{row[1]},{row[2]}\n")

    with open(f"{output_prefix}_Biological_Process_Predictions.csv", "a") as bp_file:
        bp_file.write("UniProt_ID,GO_Term,Score\n")
        for row in biological_process_data:
            bp_file.write(f"{row[0]},{row[1]},{row[2]}\n")


def chunk_sequences(sequences, size=20):
    for i in range(0, len(sequences), size):
        yield sequences[i:i + size]

def extract_sequences_from_fasta(fasta_file):
    """
    Extract sequences from FASTA content and return them in a format ready to be sent.
    """
    formatted_sequences = []
    current_sequence = ''
    current_header = ''

    # Split the content by lines
    lines = open(fasta_file, 'r').read().strip().split('\n')
    
    for line in lines:
        if line.startswith('>'):
            # If we reach a new header and current_sequence is not empty, save the sequence
            if current_sequence:
                formatted_seq = f">{current_header}\n{current_sequence}"
                formatted_sequences.append(formatted_seq)
                current_sequence = ''

            # Extract UniProt ID from the header
            current_header = line.split()[0][1:]
        else:
            current_sequence += line

    # Save the last sequence if there is one
    if current_sequence:
        formatted_seq = f">{current_header}\n{current_sequence}\n"
        formatted_sequences.append(formatted_seq)
    #import ipdb; ipdb.set_trace() 
    return formatted_sequences


def main(input_file, output_prefix):
    sequences = extract_sequences_from_fasta(input_file)
    open(f"{output_prefix}_Molecular_Function_Predictions.csv", 'w').close()
    open(f"{output_prefix}_Biological_Process_Predictions.csv", 'w').close()
    open(f"{output_prefix}_Cellular_Component_Predictions.csv", 'w').close()
    wait_time = 1 # seconds to wait between requests
    for batch in chunk_sequences(sequences):
        fasta_string = "\n".join(batch)
        successful = False
        tries = 0
        while not successful:
            tries += 1
            if tries > 1:
                print('Number of tries:' + str(tries))
                print('Fasta uploaded that is not getting successful response:')
                print(fasta_string)
            response = predict_function_with_deepgoplus(fasta_string)
            successful = handle_deepgoplus_response(response, output_prefix)
            time.sleep(wait_time)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_prefix = "test_proteins"
    main(input_file, output_prefix)

