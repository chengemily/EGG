#!/bin/bash
python -m egg.zoo.imitation_learning.imitation_pressure \
  --n_values 10 \
  --n_attributes 2 \
  --vocab_size 100 \
  --max_len 3 \
  --batch_size 512 \
  --sender_cell gru \
  --receiver_cell gru \
  --hidden 500 \
  --n_epochs 20 \
  --n_turns 50 \
  --loss kl \
  --save False \
  --kick imitation \
  --early_stopping_thr 1.1 \
  --turn_taking fixed \
  --tensorboard False \
  --checkpoint_dir checkpoints/imitation/ 

