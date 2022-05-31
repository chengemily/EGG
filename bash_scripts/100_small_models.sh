#!/bin/bash
python -m egg.zoo.compo_vs_generalization.train \
  --n_values 10 50 \
  --n_attributes 2 5 10\
  --vocab_size 50 100 1000 \
  --max_len 3 10 \
  --batch_size 5120 \
  --sender_cell gru \
  --receiver_cell gru \
  --hidden 50 100 500 \
  --n_epochs 5000 \
  --save True \
  --checkpoint_freq 0 \
  --early_stopping_thr 0.9 \ 
  #--tensorboard
