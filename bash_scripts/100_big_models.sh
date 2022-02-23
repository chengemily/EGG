#!/bin/bash
python -m egg.zoo.compo_vs_generalization.train \
  --n_values=10 \
  --n_attributes=3 \
  --vocab_size=100 \
  --max_len=10 \
  --batch_size=5120 \
  --sender_cell=gru \
  --receiver_cell=gru \
  --sender_hidden=500 \
  --receiver_hidden=500 \
  --n_epochs=5000 \
  --save=True \
  --checkpoint_dir=checkpoints/big/ \
  --checkpoint_freq=300 \
  --early_stopping_thr=0.9 
  #--tensorboard

