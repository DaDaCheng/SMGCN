#! /bin/sh
#
# This script is to reproduce our results in Table 2.
for name in cora ncora texas chameleon csbm
do
        for net in 0 1 2 3
        do
                for seed in 2 3 4 5 6 7 8 9 10
                #for seed in 1
                do      
                        for rate in 0.01 0.02 0.04 0.08 0.16 0.32 0.64
                        do
                                sbatch ./bash/run.sh python -u main.py --net $net --name $name --seed $seed --rate $rate
                        done
                done
        done
done

