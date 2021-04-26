#!/bin/bash -l

module load libs/nvidia-cuda/10.1.168/bin

nvcc --version

nvcc main.cu -std=c++11 -L/usr/local/cuda/lib64 -lcudart -o main.exe \
	&& ./main.exe ./Data/100Nodes.txt 10000
