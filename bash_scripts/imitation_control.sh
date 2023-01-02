#!/bin/bash
<<<<<<< HEAD
#MSUB -a gen13547
#MSUB -q v100
#MSUB -n 3
#MSUB -c 10
#MSUB -m scratch,work,store
#MSUB -o out_control
#MSUB -e err_control
#MSUB -T 21000

module purge
module load pytorch/1.8.0

python3 -m egg.zoo.imitation_learning.imitation_pressure \
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
  --n_turns 80 \
  --save True \
  --early_stopping_thr 1.1 \
  --ablation all \
  --kick none \
  --loss cross_entropy \
  --turn_taking fixed \
  --tensorboard True \
  --tensorboard_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/runs/imitation/control/ \
  --checkpoint_dir /ccc/scratch/cont003/gen13547/chengemi/EGG/checkpoints/imitation/control/ \
  --random_seed 1 \
  --imitation_weight 0.0
=======
python -m egg.zoo.imitation_learning.imitation_pressure \
  --n_values 10 \
  --n_attributes 2 \
  --vocab_size 100 \
  --max_len 3 \
  --batch_size 512 \
  --sender_cell gru \
  --receiver_cell gru \
  --hidden 500 \
  --n_epochs 50 \
  --n_turns 20 \
  --save True \
  --kick none \
  --turn_taking fixed \
  --early_stopping_thr 1.1 \
  --tensorboard True \
  --tensorboard_dir runs/imitation/control/ \
  --checkpoint_dir checkpoints/imitation/control/

>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
