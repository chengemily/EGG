#!/bin/bash
#MSUB -a gen13547
#MSUB -q v100
#MSUB -n 1
#MSUB -c 10
#MSUB -m scratch,work,store
#MSUB -o out_control
#MSUB -e err_control
#MSUB -T 3600

module purge
module load pytorch/1.8.0

python3 -m egg.zoo.imitation_learning.mixture_senders \
  --n_values 10 \
  --cuda True \
  --n_attributes 6 \
  --vocab_size 10 \
  --max_len 6 \
  --batch_size 1024 \
  --sender_cell gru \
  --receiver_cell gru \
  --lr 0.005 \
  --hidden 128 \
  --sender_emb 128 \
  --receiver_emb 128 \
  --n_epochs 50 \
  --n_turns 80 \
  --save 1 \
  --early_stopping_thr 1.1 \
  --loss cross_entropy \
  --tensorboard True \
  --tensorboard_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/runs/imitation/control/ \
  --checkpoint_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/checkpoints/imitation/control/ \
  --random_seed 0 \
  --entropy_weight 0.01 \
  --reinforce 0 \
  --experts Compositional Noncompositional \
  --n_epochs 100
