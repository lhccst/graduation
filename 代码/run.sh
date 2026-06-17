ALL=("tongxin" )
ALPHA_VALUES=(0.1)
DISTRIBUTION=("dirichlet")
DATASETS=("kronoDroid")
# ALGORITHMS=( "FedAvg" "FedProx" "FedPHP" "FedAMP" "pFedHN"  "APPLE" "FedGH" "FedDistill""MOON" "FedProto" "FPL" "DBE" "FedTGP" "FedCPCL")
# ALGORITHMS=( "pFedHN" "FedDistill" "FedCPCL")
ALGORITHMS=("FedMRL" "FedSSA")
# ALGORITHMS=( "FedMRL")
# ALGORITHMS=( "FedMRL" )
# ALGORITHMS=( "LG" )
# ALGORITHMS=("FedGen" "FedMRL")
# PREFIX="htfl2-321-2"
# MODEL_NAME="LeNet5"
NUM_CLIENTS_VALUES=(20)
# YAML_FILE="./configs/FedRepAPA.yaml"
LOWEST_JOIN_RATE=(0.6)
NUM_REPEAT_TIMES=(1)
debugFlag=("False")
DEBUG=("debug")
FLTYPE=("HtFL")
# LAMBDA=(0)
# 生成并运行命令

for i in "${ALL[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for algorithm in "${ALGORITHMS[@]}"; do
             for num_clients in "${NUM_CLIENTS_VALUES[@]}"; do
                 for distribution in "${DISTRIBUTION[@]}"; do
                      for alpha in "${ALPHA_VALUES[@]}"; do
                          # for lam in "${LAMBDA[@]}"; do
                            YAML_FILE="./configs/${algorithm}.yaml"
                            PREFIX="49exp-${i}"
                            echo "Before modification:"
                            cat $YAML_FILE
                             sed -i "s/^alpha: .*/alpha: $alpha/" $YAML_FILE
                             sed -i "s/^distribution: .*/distribution: $distribution/" $YAML_FILE
                            sed -i "/^client:/,/^  learning_rate:/s/^  num_clients: .*/  num_clients: $num_clients/" $YAML_FILE
                            sed -i "s/^num_repeat_times: .*/num_repeat_times: $NUM_REPEAT_TIMES/" $YAML_FILE
                            sed -i "/^server:/,/^client:/s/^  lowest_join_rate: .*/  lowest_join_rate: $LOWEST_JOIN_RATE/" $YAML_FILE
                            # sed -i "/^server:/,/^client:/s/^  lambdaWeight: .*/  lambdaWeight: $lam/" $YAML_FILE


                                            # 动态调整 num_classes_per_client 参数
                            if [[ "$distribution" == "non-balanced" && "$dataset" == "cifar100" ]]; then
                                num_classes_per_client=10
                            elif [[ "$distribution" == "non-balanced" ]]; then
                                num_classes_per_client=2
                            else
                                num_classes_per_client=2
                            fi

                            # if [[  "$dataset" == "cifar100" ]]; then
                            #     alpha=0.01
                            # else
                            #     alpha=0.1
                            # fi

                            sed -i "s/^alpha: .*/alpha: $alpha/" $YAML_FILE
                            # 修改 YAML 文件中的 num_classes_per_client 参数
                            sed -i "/^client:/,/^  learning_rate:/s/^  num_classes_per_client: .*/  num_classes_per_client: $num_classes_per_client/" $YAML_FILE

                            echo "After modification:"
                            cat $YAML_FILE
                             if [[  "$debugFlag" == "True" ]]; then
                                DEBUG="--debug"
                            else
                                DEBUG=""
                            fi

                            if [[  "$dataset" == "kronoDroid" ]]; then
                                command="python main.py --dataset $dataset --algorithm $algorithm --prefix $PREFIX  --benignAlone $DEBUG --flType $FLTYPE"
                            else
                              command="python main.py --dataset $dataset --algorithm $algorithm --prefix $PREFIX $DEBUG --flType $FLTYPE"
                            fi
                            echo "Running command: $command"
                            eval $command
                        # done
                    done
               done
            done
            done
        done
done

# ALL=("1"  )
# ALPHA_VALUES=(0.1)
# DISTRIBUTION=("dirichlet")
# DATASETS=("kronoDroid")
# # ALGORITHMS=( "FedAvg" "FedProx" "FedPHP" "FedAMP" "pFedHN"  "APPLE" "FedGH" "FedDistill""MOON" "FedProto" "FPL" "DBE" "FedTGP" "FedCPCL")
# # ALGORITHMS=( "pFedHN" "FedDistill" "FedCPCL")
# # ALGORITHMS=("FedPSYSSALong" "FedDistill")
# ALGORITHMS=( "FedCPCL")
# # ALGORITHMS=( "FedMRL" )
# # ALGORITHMS=( "LG" )
# # ALGORITHMS=("FedGen" "FedMRL")
# # PREFIX="htfl2-321-2"
# # MODEL_NAME="LeNet5"
# NUM_CLIENTS_VALUES=(20)
# # YAML_FILE="./configs/FedRepAPA.yaml"
# LOWEST_JOIN_RATE=(0.6)
# NUM_REPEAT_TIMES=(1)
# debugFlag=("True")
# DEBUG=("debug")
# FLTYPE=("no")
# # LAMBDA=(0)
# # 生成并运行命令

# for i in "${ALL[@]}"; do
#     for dataset in "${DATASETS[@]}"; do
#         for algorithm in "${ALGORITHMS[@]}"; do
#              for num_clients in "${NUM_CLIENTS_VALUES[@]}"; do
#                  for distribution in "${DISTRIBUTION[@]}"; do
#                       for alpha in "${ALPHA_VALUES[@]}"; do
#                           # for lam in "${LAMBDA[@]}"; do
#                             YAML_FILE="./configs/${algorithm}.yaml"
#                             PREFIX="htfl2-323-20-${i}"
#                             echo "Before modification:"
#                             cat $YAML_FILE
#                              sed -i "s/^alpha: .*/alpha: $alpha/" $YAML_FILE
#                              sed -i "s/^distribution: .*/distribution: $distribution/" $YAML_FILE
#                             sed -i "/^client:/,/^  learning_rate:/s/^  num_clients: .*/  num_clients: $num_clients/" $YAML_FILE
#                             sed -i "s/^num_repeat_times: .*/num_repeat_times: $NUM_REPEAT_TIMES/" $YAML_FILE
#                             sed -i "/^server:/,/^client:/s/^  lowest_join_rate: .*/  lowest_join_rate: $LOWEST_JOIN_RATE/" $YAML_FILE
#                             # sed -i "/^server:/,/^client:/s/^  lambdaWeight: .*/  lambdaWeight: $lam/" $YAML_FILE


#                                             # 动态调整 num_classes_per_client 参数
#                             if [[ "$distribution" == "non-balanced" && "$dataset" == "cifar100" ]]; then
#                                 num_classes_per_client=10
#                             elif [[ "$distribution" == "non-balanced" ]]; then
#                                 num_classes_per_client=2
#                             else
#                                 num_classes_per_client=2
#                             fi

#                             if [[  "$dataset" == "cifar100" ]]; then
#                                 alpha=0.01
#                             else
#                                 alpha=0.1
#                             fi

#                             sed -i "s/^alpha: .*/alpha: $alpha/" $YAML_FILE
#                             # 修改 YAML 文件中的 num_classes_per_client 参数
#                             sed -i "/^client:/,/^  learning_rate:/s/^  num_classes_per_client: .*/  num_classes_per_client: $num_classes_per_client/" $YAML_FILE

#                             echo "After modification:"
#                             cat $YAML_FILE
#                              if [[  "$debugFlag" == "True" ]]; then
#                                 DEBUG="--debug"
#                             else
#                                 DEBUG=""
#                             fi

#                             if [[  "$dataset" == "kronoDroid" ]]; then
#                                 command="python main.py --dataset $dataset --algorithm $algorithm --prefix $PREFIX  --benignAlone $DEBUG --flType $FLTYPE"
#                             else
#                               command="python main.py --dataset $dataset --algorithm $algorithm --prefix $PREFIX $DEBUG --flType $FLTYPE"
#                             fi
#                             echo "Running command: $command"
#                             eval $command
#                         # done
#                     done
#                done
#             done
#             done
#         done
# done
