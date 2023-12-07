# Protein-Vec: Repo for the mixture of experts model, Protein-vec

Here are instructions for how to use Protein-Vec.

First, install the GitHub repository as follows:
> git clone https://github.com/tymor22/protein-vec.git

Install required packages, run from within the protein-vec directory:

> pip install .

> pip install seaborn faiss-gpu jupyter notebook

# Download Protein-Vec mixture of experts model and each of the Aspect-Vec (expert) models

Now download all of the aspect-vec and the protein-vec models with the following command (approximately ~3GB in total): 
> wget https://users.flatironinstitute.org/thamamsy/public_www/protein_vec_models.gz

Unzip this directory of models with the following command:
> tar -zxvf protein_vec_models.gz

Now move this directory of models into the same directory as ‘src_run’ that you just installed using git clone. There are relative paths so it is important that it is moved there.

> mv protein_vec_models protein-vec/src_run/

# Download Protein-Vec lookup database
In order to perform Protein-Vec search, you will need to read from a Protein-Vec lookup database. Download this lookup database and the corresponding metadata with the following command:
> wget https://users.flatironinstitute.org/thamamsy/public_www/protein_vec_embeddings.gz

Now unzip it with the following command: 
> tar -zxvf protein_vec_embeddings.gz

Move this into the src_run directory as well.

> mv protein_vec_embeddings protein-vec/src_run/

# Tutorial
To follow an instructional tutorial of how to use Protein-Vec, follow along with the notebook: “gh_encode_and_search_new_proteins.ipynb” which is in the src_run directory.

In this notebook, you will learn how to encode proteins using Protein-Vec, and visualize/cluster those proteins. You will also learn how to search using Protein-Vec.

For a dataset with sequences and other meta data fields to follow the tutorial notebook, you can download the uniprot data as follows:
> wget https://users.flatironinstitute.org/thamamsy/public_www/uniprotkb_AND_reviewed_true_2023_07_03.tsv


Create the directory data/ in the src_run/ directory, and move the dataset file to it:
> mkdir src_run/data/

> mv uniprotkb_AND_reviewed_true_2023_07_03.tsv src_run/data/

