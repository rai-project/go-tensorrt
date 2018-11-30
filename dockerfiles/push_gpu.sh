#! /bin/bash

make docker_push_gpu

while [ $? -ne 0 ]; do
  make docker_push_gpu
done
