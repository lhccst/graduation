""" FedCPCL 用图 """
import os

import matplotlib.pyplot as plt

# 设置中文字体为宋体
import ujson

plt.rcParams['font.family'] = 'STFangsong'


num_clients = 20
algorithms = ['LG','FedGen','FedGH','FedProto', 'FedKDDC','FedTGP','FedSSA','FedIADA']
markers = ['o', 's', '^', '>', '<', 'p', '*', 'x', '+']
# algorithms = ["FedPSYSSALong"]
# markers = ['o']


communication_rounds = list(range(0, 50))  # 生成轮次列表
plt.figure(figsize=(10, 13))

for index, algorithm in enumerate(algorithms):
    # 读取客户端
    filename = "HtFE4-" + algorithm + "-" + str(num_clients) + "-0.1.json"
    path = os.path.join("G:\联邦\新建文件夹\图片\shoulian", filename)
    with open(path, "r") as f:
        result = ujson.load(f)
        if algorithm == "FedCPCL":
            best_index = result["best_acc"].index(max(result["best_acc"]))
        else:
            best_index = result["best_acc"].index(min(result["best_acc"]))
        acc_record = result["experiment"][best_index]["acc_record"]
        accuracy = [acc_record[str(round)] for round in communication_rounds]
        plt.plot(communication_rounds, accuracy, label=algorithm, marker=markers[index])

# 添加图例、标题和坐标轴标签
plt.legend()
# plt.title('CIFAR-10数据集实际数据分布场景中各算法的Top-1准确度随通信轮次的变化(20个用户)')
plt.xlabel('通信轮次', fontsize=16)
plt.ylabel('准确度', fontsize=16)
plt.grid(True)
plt.show()
