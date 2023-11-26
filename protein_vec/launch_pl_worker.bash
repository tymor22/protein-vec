#!/usr/bin/env bash
#SBATCH -N4
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

export DATA_NAME=moe_all
#Dataset path
export METAG_DATA=/mnt/home/thamamsy/ceph/protein_vec/data/uniprot_data/training_splits/combos_training_data.parquet
#Emnedding folder (pickle files)
export METAG_EMBEDDING=/mnt/home/thamamsy/ceph/swiss/swiss_protrans/pickles

export METAG_DMODEL=512
export METAG_NLAYER=2
export METAG_NHEADS=4
export METAG_IN_DIM=2048
export METAG_WARMUP_STEPS=500
export METAG_TRAIN_PROP=0.95 
export METAG_VAL_PROP=0.005
export METAG_TEST_PROP=0.005

set -ex
export METAG_LR=0.0001
export METAG_BSIZE=16 #16 for most
export METAG_SESSION=/mnt/home/thamamsy/ceph/protein_vec/models/model${METAG_LR}_dmodel${METAG_DMODEL}_nlayer${METAG_NLAYER}_${DATA_NAME}
export EPOCHS=5
export NCCL_DEBUG=INFO

set +x

srun pl_worker.bash
