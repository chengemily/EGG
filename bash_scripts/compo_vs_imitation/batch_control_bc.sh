#!/bin/bash

for rs in {0..30}
do
    export RS=$rs
    for bc_rs in {0..2}
    do
        export BCRS=$bc_rs
        export FILE="/ccc/scratch/cont003/gen13547/chengemi/EGG/bc_checkpoints/bc_randomseed_${BCRS}_from_randomseed_${RS}_checkpoint.tar" 
        if [[ ! -f "$FILE" ]]; then
            echo "$FILE not exists."
            ccc_msub ./bash_scripts/compo_vs_imitation/bc.sh
        fi
        #ccc_msub ./bash_scripts/compo_vs_imitation/bc.sh
    done
done
