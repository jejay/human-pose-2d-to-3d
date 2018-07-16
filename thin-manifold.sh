#!/bin/bash
#$ -cwd
#$ -pe gpu-titanx 1
#$ -l h_vmem=20G
. /etc/profile.d/modules.sh
module load anaconda
source activate tensorflow
python run.py \
--batch_size 64 \
--window_length 256 \
--num_manifolds 1 \
--manifold_depth 1 \
--filter_length 21 \
--conv_channels 256 \
--nointermediate_supervision \
--intra_residual \
--nointer_residual \
--maxpooling \
--noaccurate \
--ou 0.0 \
--nn 0.1 \
--mask 0.2 \
--fldout \
--data_dir /exports/eddie/scratch/s1792338/data \
--train_dir /exports/eddie/scratch/s1792338/experiments \
--dataset h36m
