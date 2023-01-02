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
  --n_epochs 50 \
  --n_turns 20 \
  --save True \
  --kick imitation \
  --early_stopping_thr 1.1 \
  --turn_taking fixed \
  --tensorboard True \
  --checkpoint_dir checkpoints/imitation/fixed/ \
  --tensorboard_dir runs/imitation/fixed/n_epochs_50

