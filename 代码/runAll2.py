import json
import os
import numpy as np
import math
import ujson


# 使用示例
if __name__ == "__main__":
    # 调用函数处理图片中的文件夹
    num_clients = 20
    model_family = 'HtFE4'
    algorithms = ['LG','FedGen','FedGH','FedProto', 'FedDistill','FedTGP','FedSSA','pFedAFM','FedRAL','FedPSYSSALong','FedSC']
    # model_families = ['HtFE2', 'HtFE4', 'HtFE5']
    datasets = ['fmnist','cifar10','cifar100']
    # ditribution = 'non-balanced'
    ditribution = 'dirichlet'

    for dataset in datasets:
        print(f"====={dataset}=====")
        for index, algorithm in enumerate(algorithms):
            if dataset == "fmnist" or dataset == "cifar10":
                filename = model_family + "-" + algorithm + "-" + str(num_clients) + "-0.1" + ".json"
            if dataset == "cifar100":
                filename = model_family + "-" + algorithm + "-" + str(num_clients) + "-0.01" + ".json"
            # path = os.path.join("finalResults", "cifar10", "non-balanced", filename)
            results_path = r"G:\联邦\新建文件夹\实验结果\results"
            results_path = r"C:\Users\Lenovo\Desktop\finalResults"
            # results_path = r"./results"

            path = os.path.join(results_path, dataset, ditribution, filename)

            if not os.path.exists(path):
                print(f"[{algorithm:}] 文件不存在: {filename}")
                continue

            with open(path, "r") as f:
                result = ujson.load(f)
                if algorithm == "FedSC" or algorithm == "FedPSYSSALong" :
                    best_index = result["best_acc"].index(max(result["best_acc"]))
                    # print(f"{filename}: {(result['best_acc'])}  {result['mean']:.4f}")
                    print(f"{filename}: {max(result['best_acc']):.4f}  {result['mean']:.4f} {(result['best_acc'])} ")

                else:
                    best_index = result["best_acc"].index(min(result["best_acc"]))
                    # print(f"{filename}: {(result['best_acc'])}  {result['mean']:.4f}")
                    print(f"{filename}: {min(result['best_acc']):.4f}  {result['mean']:.4f} {(result['best_acc'])} ")
        # ===================================================通信量
        # if dataset == 'cifar10':
        #     for index, algorithm in enumerate(algorithms):
        #         if dataset == "fmnist" or dataset == "cifar10":
        #             filename = model_family + "-" + algorithm + "-" + str(num_clients) + "-0.1" + ".json"
        #
        #         results_path = r"C:\Users\Lenovo\Desktop\finalResults"
        #         # results_path = r"./results"
        #
        #         path = os.path.join(results_path, dataset, ditribution, filename)
        #
        #         if not os.path.exists(path):
        #             print(f"[{algorithm:}] 文件不存在: {filename}")
        #             continue
        #
        #         with open(path, "r") as f:
        #             result = ujson.load(f)
        #         print(f"====={path}=====")
        #         best_index = result["best_acc"].index(max(result["best_acc"]))
        #         print(f"{filename}: {max(result['best_acc']):.4f} ")
        #         print(f"allTimeMean: {(result['allTimeMean']/60):.2f} min ")
        #         print(f"iterationTimeMean: {(result['iterationTimeMean']):.2f} sec")
        #         print(f"iterationTimeMean: {(result['memoryMean'])} ")
        #         upload = parse_value_with_unit(result['uploadMean'] )
        #         download = parse_value_with_unit(result['downloadMean'] )
        #         tongxin = (upload + download) / 16 * 20 / 1000 * 1024
        #         print(f"upload + download: {tongxin:.2f} KB")
        #
        #
        # def parse_value_with_unit(value_str):
        #     """解析带单位的数值字符串，如 '6.54 MB'"""
        #     # 分割字符串，取数字部分
        #     num_part = value_str.split()[0]  # 得到 '6.54'
        #     return float(num_part)
# ====================================
