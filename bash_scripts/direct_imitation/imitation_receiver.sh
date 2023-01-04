#!/bin/bash
#MSUB -a gen13547
#MSUB -q v100l
#MSUB -n 1
#MSUB -c 10
#MSUB -m scratch,work,store
#MSUB -o out_control
#MSUB -e err_control
#MSUB -T 80000
#MSUB -Q normal

#module purge
#module load pytorch/1.8.0

python3 -m egg.zoo.imitation_learning.direct_imitation_pressure \
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
  --n_epochs 50 \
  --n_turns 100 \
  --save True \
  --early_stopping_thr 1.1 \
  --ablation receiver_only \
  --kick imitation \
  --imitation_reinforce True \
  --loss cross_entropy \
  --turn_taking fixed \
  --population_size 2 \
  --validation_freq 50 \
  --tensorboard True \
  --sample_pairs False \
  --tensorboard_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/runs/direct_imitation/receiver_reinforce/ \
  --checkpoint_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/checkpoints/direct_imitation/ \
  --random_seed 0 \
  --imitation_weight 0.001