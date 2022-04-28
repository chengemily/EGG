#!/bin/bash
lrs=(0.1 0.05 0.01 0.005);


for L in ${lrs[@]}; do
  echo $L
	python train_comm_game.py --ablation student-no-teacher --train_set one --learning_rate "$L" &
done
wait
