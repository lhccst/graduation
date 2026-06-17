import os
import yaml
# 定义命令参数
model_families = ['HtFE5']
prefix = 'HtFE2-pic'
datasets = ['cifar100']
# algorithms = ['pFedAFM','FedRAL','FedPSYSSALong','LG','FedGen','FedGH','FedProto', 'FedDistill','FedTGP','FedSSA','pFedAFM','FedRAL','FedPSYSSALong','FedSC']
# algorithms = ['LG','FedGen','FedGH','FedProto', 'FedDistill','FedTGP','FedSSA','pFedAFM','FedRAL']
# algorithms = ['FedPSYSSALong'],'FedMRL'
algorithms = ['FedSC','LG']
# algorithms = ['pFedAFM','FedRAL']


# model_name = 'LeNet5'
# benign_alone = '--benignAlone'

config_file_root = f'./configs/'

# 生成并运行命令
for dataset in datasets:
    for model_family in model_families:
        for algorithm in algorithms:
            config_file_name = os.path.join(config_file_root, f"{algorithm}.yaml")
            with open(config_file_name, 'r', encoding='utf-8') as f:            # print(config_file_name)

                config = yaml.load(f, Loader=yaml.FullLoader)
            config['num_repeat_times'] = 3
            config['distribution'] = 'dirichlet'
            if algorithm == 'FedSC':
                config['server']['lowest_join_rate'] = 1
            else:
                config['server']['lowest_join_rate'] = 0.6
                config['server']['global_rounds'] =50

            config['client']['num_clients'] = 20
            if dataset == 'cifar100':
                config['alpha'] = 0.01
                config['client']['num_classes_per_client'] = 10
            else:
                config['alpha'] = 0.1
                config['client']['num_classes_per_client'] = 2
            # 将修改后的对象写回 YAML 文件
            with open(config_file_name, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, indent=2, sort_keys=False)
            prefix = model_family + "-1"
            command = f"python main.py --dataset {dataset} --algorithm {algorithm} --prefix {prefix} --model_family {model_family} "
            print(f"Running command: {command}")
            os.system(command)
