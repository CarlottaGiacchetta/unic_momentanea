#!/bin/bash

GPU_ID=2;

docker run --rm --runtime=nvidia --name='class-cgiacchetta-rsde-'${GPU_ID} -e CUDA_VISIBLE_DEVICES=$GPU_ID --ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 -t --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) class-cgiacchetta-rsde2:latest 
