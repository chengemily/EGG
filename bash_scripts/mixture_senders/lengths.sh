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
  --cuda False \
  --n_attributes 2 \
  --vocab_size 10 \
  --max_len 6 \
  --batch_size 1024 \
  --sender_cell gru \
  --receiver_cell gru \
  --lr 0.005 \
  --hidden 128 \
  --sender_emb 128 \
  --receiver_emb 128 \
  --save 1 \
  --loss cross_entropy \
  --tensorboard True \
  --tensorboard_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/runs/imitation/control/ \
  --checkpoint_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/checkpoints/imitation/control/ \
  --random_seed 0 \
  --entropy_weight 0.01 \
  --reinforce 1 \
  --experts VariableLength VariableLength VariableLength \
  --expert_lengths 2 4 6 \
  --variable_message_length 1 \
  --n_epochs 200
