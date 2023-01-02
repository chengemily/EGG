#!/bin/bash

for rs in {0..30}
do
    export RS=$rs
    ccc_msub ./bash_scripts/compo_vs_imitation/train.sh
done
