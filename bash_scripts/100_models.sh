#!/bin/bash
python -m egg.zoo.compo_vs_generalization.train \
  --n_values=10 \
  --n_attributes=2 \
  --vocab_size=100 \
  --max_len=3 \
  --batch_size=5120 \
  --sender_cell=gru \
  --receiver_cell=gru \
  --sender_hidden=500 \
  --receiver_hidden=500 \
  --n_epochs=3000 \
  --save=True \
  --checkpoint_dir=checkpoints/ \
  --checkpoint_freq=300 \
  --early_stopping_thr=0.9 
  #--tensorboard

