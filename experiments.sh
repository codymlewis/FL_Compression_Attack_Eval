#!/usr/bin/env bash

for agg in fedavg contra flame foolsgold krum std_dagmm viceroy; do
    for comp in fedmax fedprox fedzip ae; do
        for attack in onoff labelflip; do
            for dataset in mnist kddcup99; do
                for aper in 0.3 0.5; do
                    python main.py --agg "$agg" --comp "$comp" --attack "$attack" --dataset "$dataset" --aper "$aper"
                done
            done
        done
    done
done