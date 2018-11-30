#! /bin/bash

make docker_push_cpu

while [ $? -ne 0 ]; do
  make docker_push_cpu
done
