#!/bin/bash

for rs in {0..12}
do
    export RS=$rs
    ccc_msub ./bash_scripts/imitation_control_simul.sh

    for iw in 0.025 0.05 0.075
    do
        export IW=$iw
        ccc_msub ./bash_scripts/imitation_simul_rcvr_only.sh
        ccc_msub ./bash_scripts/imitation_simul_sender_only.sh
    done
done
