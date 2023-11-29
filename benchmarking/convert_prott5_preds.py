import pandas as pd
import numpy as np
import sys

def prott5_to_deepgoplus_format(prott5_file, output_file, alpha=1):
    # Read the Prott5 file
    prott5_df = pd.read_csv(prott5_file)

    # Initialize an empty dataframe for the output
    output_df = pd.DataFrame(columns=['UniProt_ID', 'GO_Term', 'Score'])

    # Iterate through each row in the Prott5 dataframe
    for index, row in prott5_df.iterrows():
        # Extract the necessary information
        protein_id = row['original_id']
        annotations_1 = row['k_nn_1_annotations'].split(';')
        annotations_2 = row['k_nn_2_annotations'].split(';')
        annotations_3 = row['k_nn_3_annotations'].split(';')

        '''
        similarity_1 = 1 - row['k_nn_1_distance']
        similarity_2 = 1 - row['k_nn_2_distance']
        similarity_3 = 1 - row['k_nn_3_distance']
        '''
        # Convert Euclidean distance to similarity
        similarity_1 = np.exp(-alpha * row['k_nn_1_distance'])
        similarity_2 = np.exp(-alpha * row['k_nn_2_distance'])
        similarity_3 = np.exp(-alpha * row['k_nn_3_distance'])

        # Combine annotations and similarities
        annotations = annotations_1 + annotations_2 + annotations_3
        similarities = [similarity_1] * len(annotations_1) + [similarity_2] * len(annotations_2) + [similarity_3] * len(annotations_3)

        # Calculate scores for each annotation
        scores = {}
        for annotation, similarity in zip(annotations, similarities):
            if annotation not in scores:
                scores[annotation] = 0
            scores[annotation] += similarity

        # Normalize the scores
        total_score = sum(scores.values())
        for annotation in scores:
            scores[annotation] /= total_score

        # Add the scores to the output dataframe
        for annotation, score in scores.items():
            output_df = output_df.append({
                'UniProt_ID': protein_id,
                'GO_Term': annotation,
                'Score': score
            }, ignore_index=True)

    # Save the output dataframe to a file
    output_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    prott5_to_deepgoplus_format(sys.argv[1], sys.argv[2])

