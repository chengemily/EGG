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
  --kick none \
  --turn_taking fixed \
  --early_stopping_thr 1.1 \
  --tensorboard True \
  --tensorboard_dir runs/imitation/control/ \
  --checkpoint_dir checkpoints/imitation/control/

