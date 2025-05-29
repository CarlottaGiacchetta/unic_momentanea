#!/bin/bash

GPU_IDS="3,4"

docker run --rm --gpus '"device='${GPU_IDS}'"' \
  --name="class-cgiacchetta-rsde-2" \
  --shm-size=4g \
  -v $(pwd):/workspace \
  -w /workspace \
  -u $(id -u):$(id -g) \
  class-cgiacchetta-rsde2:latest \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  unicc/main_unic.py \
  --batch_size_per_gpu 128 \
  --data_dir dati \
  --arch vit_tiny \
  --saveckpt_freq 10 \
  --in_chans 12 \
  --concat True \
  --output_dir CANCELLAAAA
