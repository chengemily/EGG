#!/bin/bash
for i in {1..100}
do
	python -m egg.zoo.compo_vs_generalization.train \
		--n_values=5 \
		--n_attributes=2 \
		--vocab_size=100 \
		--max_len=3 \
		--batch_size=5120 \
		--sender_cell=gru \
		--receiver_cell=gru \
		--random_seed=$i \
		--sender_hidden=500 \
		--receiver_hidden=500 \
		--n_epochs=1000
done
