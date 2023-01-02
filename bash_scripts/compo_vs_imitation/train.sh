#!/bin/bash
#MSUB -a gen13547
#MSUB -q v100l
#MSUB -n 1
#MSUB -c 10
#MSUB -m scratch,store
#MSUB -o out
#MSUB -e err
#MSUB -T 10800
#MSUB -Q normal

module purge
module load pytorch/1.8.0

ccc_mprun python3 -m egg.zoo.compo_vs_generalization.train \
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
    --n_epochs 1000 \
    --save True \
    --early_stopping_thr 1.1 \
    --loss cross_entropy \
    --tensorboard_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/runs/ \
    --checkpoint_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/checkpoints/basic_correlations/ \
    --random_seed $RS
