#!/bin/bash

for rs in {0..4}
do
    export RS=$rs
    #ccc_msub ./bash_scripts/direct_imitation/imitation_control.sh

    for iw in 0.000001 0.00001 0.0001 0.001 0.01 0.1
    do
        export IW=$iw
        export PAIRS=False
        for reinforce in True False
        do
            export REINF=$reinforce
            ccc_msub ./bash_scripts/direct_imitation/imitation_receiver.sh
            ccc_msub ./bash_scripts/direct_imitation/imitation_sender.sh 
        done
    done
done
