#!/bin/bash
#MSUB -a gen13547
#MSUB -q v100
#MSUB -n 1
#MSUB -c 20
#MSUB -m scratch,store
#MSUB -o out
#MSUB -e err
#MSUB -T 10800
#MSUB -Q normal

module purge
module load pytorch/1.8.0

export RS=0
export BCRS=0

python3 -m egg.zoo.imitation_learning.behavioural_cloning \
    --n_values 10 \
    --cuda True \
    --n_attributes 6 \
    --vocab_size 10 \
    --max_len 10 \
    --batch_size 1024 \
    --holdout_density 0.99 \
    --sender_cell gru \
    --receiver_cell gru \
    --data_scaler 1 \
    --lr 0.005 \
    --hidden 128 \
    --sender_emb 128 \
    --receiver_emb 128 \
    --n_epochs_bc 2000 \
    --save_bc True \
    --early_stopping_thr_bc 1.0 \
    --sender_reward accuracy \
    --expert_seed $RS \
    --bc_random_seed $BCRS \
    --imitation_reinforce True
