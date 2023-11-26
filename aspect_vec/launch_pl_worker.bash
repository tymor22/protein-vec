#!/usr/bin/env bash
#SBATCH -N8
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH --exclusive
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem 800gb
#SBATCH --gpus-per-node=4
#SBATCH --job-name=train
#SBATCH -o slurm-%x.%j.out


source /mnt/home/thamamsy/projects/protein_vec/lib/environment/protein_vec_env/bin/activate

export METAG_NNODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
echo "Number of nodes: $METAG_NNODES"

#######################EC
#export DATA_NAME=EC_only_model
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/EC_benchmarks/EC_data/training_data_ec1234_leaving_out_20seqid_50seqid_new.parquet
#export DATA_NAME=EC4_only_model
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/EC_benchmarks/EC_data/training_data_ec4_only_leaving_out_20seqid_50seqid_new.parquet

#export DATA_NAME=EC4_only_model_negative_mining
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/EC_benchmarks/EC_data/ec_hard_combined_33M.parquet

#EC4_only_model_with_margins
export DATA_NAME=EC4_only_model_with_margins_4_8_12_25
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/EC_benchmarks/EC_data/ec_data_all_together_margin.parquet
export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/EC_benchmarks/EC_data/ec_data_all_together_margin_4_8_12_25.parquet

#######################GENE3D
#export DATA_NAME=GENE3D_only_model
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/GENE3D_benchmarks/training_data_GENE3D_leaving_out_20seqid_50seqid.parquet

#export DATA_NAME=GENE3D_identical_only_model
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/GENE3D_benchmarks/training_data_GENE3D_identical_leaving_out_20seqid_50seqid.parquet

#export DATA_NAME=GENE3D_negative_mining
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/GENE3D_benchmarks/combined_held_out_50_hard_26M.parquet

#######################PFAM
#export DATA_NAME=PFAM_only_model
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/PFAM/training_data_PFAM_google_cluster_split_train_dev_clean.parquet
#export METAG_EMBEDDING=/mnt/home/thamamsy/ceph/protein_vec/data/PFAM/pfam_pickles_cluster

#export DATA_NAME=PFAM_negative_mining
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/PFAM/training_data_PFAM_google_cluster_split_train_dev_clean_negative_mining_sampled_both.parquet
#export METAG_EMBEDDING=/mnt/home/thamamsy/ceph/protein_vec/data/PFAM/pfam_pickles_cluster

#######################GO
##########MFO
#export DATA_NAME=GO_MFO_only_model
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/GO_benchmarks/GO_make_data/GOGO/go_prep/mfo_triplets_28M.parquet
##########BPO
#export DATA_NAME=GO_BPO_only_model
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/GO_benchmarks/GO_make_data/GOGO/go_prep/bpo_triplets_48M.parquet
##########CCO
#export DATA_NAME=GO_CCO_only_model 
#export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/GO_benchmarks/GO_make_data/GOGO/go_prep/cco_triplets_41M.parquet

#export METAG_EMBEDDING=/mnt/home/thamamsy/ceph/protein_vec/data/GO_benchmarks/pickle_GO_embeddings
########################

export METAG_LOOKUP_SEQS=/mnt/home/thamamsy/ceph/function/csv_files_all/train_val_test/uniprot_seqs.pickle
export METAG_EMBEDDING=/mnt/home/thamamsy/ceph/swiss/swiss_protrans/pickles

export METAG_DMODEL=1024
export METAG_NLAYER=2
export METAG_NHEADS=4
export METAG_IN_DIM=2048
export METAG_WARMUP_STEPS=500
export METAG_TRAIN_PROP=0.95 
export METAG_VAL_PROP=0.005
export METAG_TEST_PROP=0.005

set -ex
export METAG_LR=0.0001
export METAG_BSIZE=16 
export METAG_SESSION=/mnt/home/thamamsy/ceph/protein_vec/models/model${METAG_LR}_dmodel${METAG_DMODEL}_nlayer${METAG_NLAYER}_${DATA_NAME}
export EPOCHS=5
export NCCL_DEBUG=INFO

set +x

srun pl_worker.bash
