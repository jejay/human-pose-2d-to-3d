#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1 # use 1 GPU
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -x landonia20

# Setup CUDA and CUDNN related paths
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate tensorflow

#python /home/${STUDENT_ID}/human-pose-2d-to-3d/run.py \
#--batch_size 64 \
#--window_length 256 \
#--num_manifolds 1 \
#--manifold_depth 1 \
#--filter_length 21 \
#--conv_channels 256 \
#--nointermediate_supervision \
#--intra_residual \
#--nointer_residual \
#--maxpooling \
#--noaccurate \
#--ou 0.0 \
#--nn 0.1 \
#--mask 0.2 \
#--fldout \
#--data_dir /home/${STUDENT_ID}/human-pose-2d-to-3d/data \
#--train_dir /home/${STUDENT_ID}/human-pose-2d-to-3d/experiments \
#--dataset h36m


#python /home/${STUDENT_ID}/human-pose-2d-to-3d/run.py \
#--batch_size 64 \
#--window_length 256 \
#--num_manifolds 1 \
#--manifold_depth 1 \
#--filter_length 21 \
#--conv_channels 256 \
#--nointermediate_supervision \
#--intra_residual \
#--nointer_residual \
#--nomaxpooling \
#--noaccurate \
#--ou 0.0 \
#--nn 0.1 \
#--mask 0.2 \
#--fldout \
#--data_dir /home/${STUDENT_ID}/human-pose-2d-to-3d/data \
#--train_dir /home/${STUDENT_ID}/human-pose-2d-to-3d/experiments \
#--dataset h36m

python /home/${STUDENT_ID}/human-pose-2d-to-3d/run.py \
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
--accurate \
--ou 0.0 \
--nn 0.1 \
--mask 0.2 \
--fldout \
--data_dir /home/${STUDENT_ID}/human-pose-2d-to-3d/data \
--train_dir /home/${STUDENT_ID}/human-pose-2d-to-3d/experiments \
--dataset h36m
