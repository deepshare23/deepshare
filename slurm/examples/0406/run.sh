#!/bin/bash

#unbalanced
FILE_NAME=/home/gajagajago/dlcm/slurm/examples/out/${x}_${y}_unbalanced
for x in timm graphsage fsdp transformer dlrm moe
do
	for y in timm graphsage fsdp transformer dlrm moe
	do
        touch $FILE_NAME
		sbatch ${x}_unbalanced.sh >> $FILE_NAME
        echo "\n" >> $FILE_NAME
		sbatch ${y}_unbalanced.sh >> $FILE_NAME
		sleep 15m
		scancel --user=gajagajago
	done
done

#balanced
FILE_NAME=/home/gajagajago/dlcm/slurm/examples/out/${x}_${y}_balanced
for x in timm graphsage fsdp transformer dlrm moe
do
	for y in timm graphsage fsdp transformer dlrm moe
	do
        touch $FILE_NAME
		sbatch ${x}_balanced.sh >> $FILE_NAME
        echo "\n" >> FILE_NAME
		sbatch ${y}_balanced.sh >> $FILE_NAME
		sleep 15m
		scancel --user=gajagajago
	done
done
