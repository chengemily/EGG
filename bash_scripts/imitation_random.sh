#!/bin/bash
python -m egg.zoo.imitation_learning.imitation_pressure \
  --n_values 10 \
  --n_attributes 2 \
  --vocab_size 100 \
  --max_len 3 \
  --batch_size 5120 \
  --sender_cell gru \
  --receiver_cell gru \
  --hidden 500 \
  --n_epochs 10 100 500 \
  --save True \
  --early_stopping_thr 0.9 \
  --kick random \
  --turn_taking fixed \
  --tensorboard False \
  --checkpoint_dir checkpoints/imitation/random/ \
#  --tensorboard_dir runs/imitation/random/

