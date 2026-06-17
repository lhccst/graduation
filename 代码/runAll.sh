#!/bin/bash

# 定义参数
datasets=("cifar100")
algorithms=("LG" "FedGen" "FedGH" "FedProto" "FedDistill" "FedTGP" "FedMRL" "FedSSA")
prefix="HtFE5"
model_family="HtFE5"
debug="--debug"

# 遍历所有组合并执行命令
for dataset in "${datasets[@]}"; do
    for algorithm in "${algorithms[@]}"; do
        command="python main.py --dataset $dataset --algorithm $algorithm --prefix $prefix --model_family $model_family $debug"
        echo "Running command: $command"
        eval "$command"
    done
done