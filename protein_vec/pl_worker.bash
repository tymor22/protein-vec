#!/usr/bin/env sh

##################################################################

NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#Source python virtual environment
source /mnt/home/thamamsy/projects/protein_vec/lib/environment/protein_vec_env/bin/activate

export NCCL_DEBUG=INFO

python train_protein_vec.py \
    --nodes ${METAG_NNODES} \
    --gpus ${NGPUS} \
    --session ${METAG_SESSION} \
    --data ${METAG_DATA} \
    --embeddings ${METAG_EMBEDDING} \
    --lr0 ${METAG_LR} \
    --max-epochs ${EPOCHS} \
    --batch-size ${METAG_BSIZE} \
    --d_model ${METAG_DMODEL} \
    --num_layers ${METAG_NLAYER} \
    --dim_feedforward ${METAG_IN_DIM} \
    --nhead ${METAG_NHEADS} \
    --warmup_steps ${METAG_WARMUP_STEPS} \
    --train-prop ${METAG_TRAIN_PROP} \
    --val-prop ${METAG_VAL_PROP} \
    --test-prop ${METAG_TEST_PROP}
