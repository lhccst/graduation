import os
import yaml
# 定义命令参数
datasets = ['cifar10']
algorithms = ['FedPSYSSALong']
weights = [0.05,0.1,0.3,0.5,0.7]
# weights = [0,0.1,0.3]
# weights = ['2','3']
prefix = ''


config_file_name = f'./configs/FedPSYSSALong.yaml'

# 生成并运行命令
for dataset in datasets:
    for algorithm in algorithms:
        for weight in weights:
            with open(config_file_name, 'r',encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            config['server']['lambdaWeight'] = weight
            config['num_repeat_times'] = 3
            # 将修改后的对象写回 YAML 文件
            with open(config_file_name, 'w',encoding='utf-8') as f:
                yaml.dump(config, f, indent=2,sort_keys=False)

            prefix = 'HtFE2-' + "lambda-" + str(config['server']['lambdaWeight'])
            # prefix = 'HtFE2-Newcontrast' + weight
            print("prefix: " ,prefix)
            print("clent-miu: " ,config['server']['lambdaWeight'])
            # print("clent-belta: " ,config['client']['belta'])
            command = f"python main.py --dataset {dataset} --algorithm {algorithm} --prefix {prefix} --model_family HtFE2 --debug "
            print(f"Running command: {command}")
            os.system(command)


