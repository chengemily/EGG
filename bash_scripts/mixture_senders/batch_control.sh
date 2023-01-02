#!/bin/bash

for rs in {0..4}
do
    export RS=$rs
    for ent in {0.0001 0.001 0.01 0.1 1.0}
    do
        for rf in {true false}
        do
            export RF=$rf
            ccc_msub ./bash_scripts/mixture_senders/compo_nocompo.sh
            ccc_msub ./bash_scripts/mixture_senders/entropy.sh
            ccc_msub ./bash_scripts/mixture_senders/length.sh
        done
    done
done
