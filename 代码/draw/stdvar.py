# # import json
# # import os
# # import math
# #
# # # 假设所有 JSON 文件都在当前目录下的 "data" 文件夹中
# # directory = "D:\\FedLearning\\毕设参考文献\\结果保存\\htfe4\\cifar10"  # 替换为你的 JSON 文件目录
# #
# # # 遍历目录中的所有 JSON 文件
# # for filename in os.listdir(directory):
# #     if filename.endswith(".json"):
# #         filepath = os.path.join(directory, filename)
# #
# #         # 读取 JSON 文件
# #         with open(filepath, "r") as f:
# #             data = json.load(f)
# #
# #         # 获取方差 (stdvar)
# #         variance = data["stdvar"]
# #
# #         # 计算标准差 (标准差 = 方差的平方根)
# #         std_deviation = math.sqrt(variance)
# #
# #         # 将标准差保存到 config 下的 true_stdvar 中
# #         if "config" not in data:
# #             data["config"] = {}  # 如果 config 不存在，创建一个空字典
# #         data["true_stdvar"] = std_deviation
# #
# #         # 将修改后的数据写回 JSON 文件
# #         with open(filepath, "w") as f:
# #             json.dump(data, f, indent=4)  # indent=4 用于美化输出
# #
# #         print(f"Processed {filename}: stdvar = {variance}, true_stdvar = {std_deviation}")
# #
# # #
# # # import numpy as np
# # # #
# # # # # 三个值
# # # values = [0.5154889653776068,
# # #           0.4939258959303503,
# # #         0.5048,]
# # #
# # # # 计算平均值
# # # mean_value = np.mean(values)
# # #
# # # # 计算标准差
# # # std_dev = np.std(values)
# # #
# # # print(f"平均值: {mean_value}")
# # # print(f"标准差: {std_dev}")
#
# ##


# import ujson
# import os
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# plt.rcParams['font.size'] = 22  # 设置所有字体的默认大小
# # 文件路径和保存路径
# base_path = "D:\\FedLearning\\毕设参考文献\\数据集\\all"
# config_files = ["0.05config.json", "0.1config.json", "0.3config.json", "0.5config.json"]
# savepath = os.path.join(base_path, "final321-2-song.pdf")
#
# # 颜色设置
# colors = ['#dab49d', '#adc178', '#93b7be', '#457b9d', '#0a9396', '#ee9b00', '#e9c46a',
#           '#fee440', '#f7e1d7', '#f07167', '#6d6875', '#bcb8b1', '#a49fd5', '#7FFF00']
#
# # 读取数据并计算统一的最大值
# max_height = 0
# data_list = []
# for file_name in config_files:
#     file_path = os.path.join(base_path, file_name)
#     with open(file_path, "r") as f:
#         config = ujson.load(f)
#         data = config["statistic"]
#         data_list.append(data)
#         for user_data in data:
#             total_height = sum(int(user_data[str(tag)]) if str(tag) in user_data else 0 for tag in range(13))
#             max_height = max(max_height, total_height)
#
# # 创建 2x2 子图
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()
#
# # 遍历数据和子图
# for idx, (data, ax) in enumerate(zip(data_list, axes)):
#     # 数据
#     labels = data
#     tags = list(range(0, 13))
#
#     # 柱在图横轴的位置
#     x = np.arange(len(labels))
#     cumulative_heights = np.zeros(len(labels))
#
#     # 绘制堆叠柱状图
#     for i, tag in enumerate(tags):
#         sample_counts = [user_data[str(tag)] if str(tag) in user_data else 0 for user_data in data]
#         ax.bar(x, sample_counts, width=0.5, label=f"类别 {tag}", color=colors[i], bottom=cumulative_heights)
#         cumulative_heights += np.array(sample_counts)
#
#     ax.set_xlabel("客户端序号")  # 调整字体大小
#     ax.set_ylabel("样本量")  # 调整字体大小
#     ax.set_xticks(x)
#     ax.set_xticklabels(np.arange(0, len(labels)), fontsize=16)  # 设置x轴字体大小
#     ax.set_ylim(0, max_height * 1.1)  # 统一纵坐标范围
#
#     # 设置子图标题放在下方
#     ax.text(0.5, -0.25, f"({chr(97 + idx)}) α={config_files[idx].replace('config.json', '')}", ha='center', va='center', transform=ax.transAxes,fontname='Arial')

# 添加全局标题
# fig.suptitle('不同配置下的KronoDroid数据集样本分布', fontsize=20)

# handles, labels = axes[0].get_legend_handles_labels()
# # 调整图例位置参数
# fig.legend(handles, labels,
#            loc='lower center',
#            bbox_to_anchor=(0.5, 0.02),  # 从0.02提升到0.06
#            ncol=7,
#            frameon=False,
#            fontsize=20,  # 设置图例字体大小
#            borderaxespad=0.5)  # 增加图例与标题间距
#
# # 重新分配画布空间
# plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # 底部空间从8%增加到12%
# plt.savefig(savepath, format='pdf')
# plt.show()




# import ujson
# import os
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# plt.rcParams['font.size'] = 25  # 设置所有字体的默认大小
# # 文件路径和保存路径
# base_path = "G:\联邦\新建文件夹\图片\数据集"
# config_files = ["Fashion-MNIST-config1.json","CIFAR-10-config1.json"]
# savepath = os.path.join(base_path, "final322-song1.pdf")
#
# # 颜色设置
# colors = ['#dab49d', '#adc178', '#93b7be', '#457b9d', '#0a9396', '#ee9b00', '#e9c46a',
#           '#fee440', '#f7e1d7', '#f07167', '#6d6875', '#bcb8b1', '#a49fd5', '#7FFF00']
#
# # 读取数据并计算统一的最大值
# max_height = 0
# data_list = []
# for file_name in config_files:
#     file_path = os.path.join(base_path, file_name)
#     with open(file_path, "r") as f:
#         config = ujson.load(f)
#         data = config["statistic"]
#         data_list.append(data)
#         for user_data in data:
#             total_height = sum(int(user_data[str(tag)]) if str(tag) in user_data else 0 for tag in range(13))
#             max_height = max(max_height, total_height)
#
# # 创建 2x2 子图
# fig, axes = plt.subplots(1, 2, figsize=(20, 10))
# axes = axes.flatten()
#
# # 遍历数据和子图
# for idx, (data, ax) in enumerate(zip(data_list, axes)):
#     # 数据
#     labels = data
#     tags = list(range(0, 10))
#
#     # 柱在图横轴的位置
#     x = np.arange(len(labels))
#     cumulative_heights = np.zeros(len(labels))
#
#     # 绘制堆叠柱状图
#     for i, tag in enumerate(tags):
#         sample_counts = [user_data[str(tag)] if str(tag) in user_data else 0 for user_data in data]
#         ax.bar(x, sample_counts, width=0.5, label=f"类别 {tag}", color=colors[i], bottom=cumulative_heights)
#         cumulative_heights += np.array(sample_counts)
#
#     ax.set_xlabel("客户端序号")  # 调整字体大小
#     ax.set_ylabel("样本量")  # 调整字体大小
#     ax.set_xticks(x)
#     ax.set_xticklabels(np.arange(0, len(labels)))  # 设置x轴字体大小
#     ax.set_ylim(0, max_height * 1.1)  # 统一纵坐标范围
#
#     # 设置子图标题放在下方
#     ax.text(0.5, -0.16, f"({chr(97 + idx)}) {config_files[idx].replace('-config1.json', ' ')}",
#             ha='center', va='center', transform=ax.transAxes,fontname='Arial')
#
# # 添加全局标题
# # fig.suptitle('不同配置下的KronoDroid数据集样本分布', fontsize=20)
#
# handles, labels = axes[0].get_legend_handles_labels()
# # 调整图例位置参数
# fig.legend(handles, labels,
#            loc='lower center',
#            bbox_to_anchor=(0.5, 0.0),  # 从0.02提升到0.06
#            ncol=5,
#            frameon=False,
#            borderaxespad=0.5)  # 增加图例与标题间距
#
# # 重新分配画布空间
# plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # 底部空间从8%增加到12%
# plt.savefig(savepath, format='pdf')
# plt.show()



# import os
#
#
# def generate_partition(num_clients, num_classes,
#                        is_dirichlet=False, is_balanced=False,
#                        alpha=0.1, num_classes_per_client=2):
#     """ 生成数据集分布 """
#     if is_dirichlet:
#         partition = np.random.dirichlet(alpha=[alpha] * num_clients, size=num_classes)
#     else:
#         count_classes_each_client = [num_classes_per_client for _ in range(num_clients)]  # 每个用户剩余的样本类数
#
#         num_clients_each_class = int(np.floor(num_clients * num_classes_per_client / num_classes))  # 一类样本被分配到的用户数
#
#         partition = np.zeros(shape=(num_classes, num_clients))
#         for cls in range(num_classes):
#             """ 添加共享同一类的用户 """
#             selected_clients = []
#             for client_id in range(num_clients):
#                 if count_classes_each_client[client_id] > 0 and len(selected_clients) < num_clients_each_class:
#                     selected_clients.append(client_id)
#                     count_classes_each_client[client_id] -= 1
#
#             if is_balanced:
#                 proportions = np.full(shape=num_clients_each_class, fill_value=1)
#             else:
#                 proportions = np.random.uniform(low=0.3, high=0.7, size=num_clients_each_class)  # 生成对应用户数个随机数
#             proportions = (proportions / np.sum(proportions)).tolist()
#
#             """ 给到对应的用户 """
#             for client_id in selected_clients:
#                 prop = proportions.pop()
#                 partition[cls][client_id] = prop
#
#     return partition
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
#
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# plt.rcParams['font.size'] = 22  # 设置所有字体的默认大小
# # 示例数据
# partition = generate_partition(20, 10,is_dirichlet=True, alpha=0.1)
#
# # 创建自定义颜色映射
# colors = ["#b5e2fa", "#012a4a"]  # 定义渐变的起止颜色
# cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
#
# # 设置图形大小
# plt.figure(figsize=(12, 6))  # 宽 12 英寸，高 6 英寸
#
# # 绘制矩阵图
# plt.imshow(partition, cmap=cmap, interpolation='nearest')  # 使用自定义颜色映射
# plt.colorbar(label="比例")  # 添加颜色条并设置标签
# plt.clim(0, 1)  # 设置颜色条范围
#
# # 设置标题和坐标轴标签
# # plt.title("Matrix Visualization: Users and Classes")
# # plt.xlabel("客户端序号")
# # plt.ylabel("类别")
# #
# # # 设置横纵坐标刻度
# plt.xticks(ticks=np.arange(20), labels=np.arange(20),fontsize=18)  # 横坐标 0-19
# plt.yticks(ticks=np.arange(10), labels=np.arange(10),fontsize=18)  # 纵坐标 0-9
# #
#
# # 修改横纵坐标字体大小
# # plt.xticks(fontsize=12)  # 修改横坐标字体大小为 12
# # plt.yticks(fontsize=12)  # 修改纵坐标字体大小为 12
#
# # 设置标题和坐标轴
# # plt.title("Matrix Visualization", fontsize=14)  # 标题字体大小
# plt.xlabel("客户端序号", fontsize=20)  # 横坐标标签字体大小
# plt.ylabel("类别", fontsize=20)  # 纵坐标标签字体大小
# # 显示图形
# savepath = os.path.join("D:\\FedLearning\\毕设参考文献\\数据集\\0.1","dis-song.pdf")
# plt.savefig(savepath, format='pdf')
# plt.show()
#
#



import os

import matplotlib.pyplot as plt
import numpy as np

# 这里是你的实验数据
# data = {
#     "Experiment_0": {
#         "model": "LG-FedAvg",
#         "GlobalClassAcc": [
#             96.8122409945808,
#             84.09191338930623,
#             72.02380952380952,
#             74.6938775510204,
#             75.91836734693878,
#             97.47899159663865,
#             0,
#             0.0,
#             36.8421052631579,
#             63.03142329020333,
#             73.33333333333333,
#             93.97590361445783,
#             9.45273631840796
#         ],
#     },
#     "Experiment_1": {
#         "model": "FedGen",
#         "GlobalClassAcc": [
#             95.72704081632654,
#             90.90106007067138,
#             71.96261682242991,
#             49.38775510204081,
#             79.67479674796748,
#             98.33333333333333,
#             0,
#             0.0,
#             84.21052631578948,
#             54.53703703703704,
#             87.5,
#             93.97590361445783,
#             17.5
#         ],
#     },
#     "Experiment_2": {
#         "model": "FedGH",
#         "GlobalClassAcc": [
#             93.1738437001595,
#             93.55123674911661,
#             67.82978723404256,
#             67.75510204081633,
#             82.04081632653062,
#             98.33333333333333,
#             0,
#             0.0,
#             63.1578947368421,
#             60.129509713228494,
#             84.61538461538461,
#             89.41176470588235,
#             38.118811881188115
#         ],
#     },
#     "Experiment_3": {
#         "model": "FedProto",
#         "GlobalClassAcc": [
#             96.936821952776,
#             92.09014582412726,
#             59.2687074829932,
#             72.76422764227642,
#             89.83739837398375,
#             97.43589743589743,
#             0,
#             0.0,
#             94.73684210526316,
#             79.0009250693802,
#             85.71428571428571,
#             96.42857142857143,
#             14.5
#         ],
#     },
#     "Experiment_4": {
#         "model": "FedDistill",
#         "GlobalClassAcc": [
#             97.67293592604399,
#             91.29474149359258,
#             76.36054421768708,
#             66.39344262295081,
#             75.50200803212851,
#             98.36065573770492,
#             0,
#             0.0,
#             52.94117647058823,
#             66.9750231267345,
#             66.66666666666667,
#             91.56626506024097,
#             30.34825870646766
#         ],
#     },
#     "Experiment_5": {
#         "model": "FedTGP",
#         "GlobalClassAcc": [
#             95.82669640012743,
#             89.93377483443709,
#             78.8445199660153,
#             68.69918699186992,
#             90.68825910931174,
#             98.36065573770492,
#             100.0,
#             0.0,
#             68.42105263157895,
#             80.57354301572617,
#             84.61538461538461,
#             93.97590361445783,
#             10.1010101010101
#         ],
#     },
#     "Experiment_6": {
#         "model": "FedMRL",
#         "GlobalClassAcc": [
#             95.78947368421052,
#             87.10247349823321,
#             85.44680851063829,
#             76.82926829268293,
#             83.06451612903226,
#             93.33333333333333,
#             0,
#             0.0,
#             68.42105263157895,
#             39.53703703703704,
#             93.33333333333333,
#             86.74698795180723,
#             46.26865671641791
#         ],
#     },
#     "Experiment_7": {
#         "model": "FedSSA",
#         "GlobalClassAcc": [
#             96.23724489795919,
#             94.03710247349824,
#             80.71672354948805,
#             70.73170731707317,
#             91.90283400809717,
#             98.34710743801652,
#             100.0,
#             0.0,
#             50.0,
#             71.23034227567068,
#             53.333333333333336,
#             97.61904761904762,
#             30.150753768844222
#         ],
#     },
#     "Experiment_8": {
#         "model": "FedWL",
#         "GlobalClassAcc": [
#             95.91966847306344,
#             93.46289752650176,
#             93.11224489795919,
#             58.94308943089431,
#             85.71428571428571,
#             98.34710743801652,
#             100.0,
#             0.0,
#             95.23809523809524,
#             61.75925925925926,
#             70.0,
#             91.76470588235294,
#             44.776119402985074
#         ],
#     },
# }
#
# # 创建图形
# fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # 创建 3x3 网格的子图
# axes = axes.flatten()  # 扁平化数组，便于索引
# colors = ['#dab49d', '#adc178', '#93b7be', '#457b9d', '#0a9396', '#ee9b00', '#e9c46a',
#           '#fee440', '#f7e1d7', '#f07167', '#6d6875', '#bcb8b1', '#a49fd5', '#7FFF00']
# #
# # 绘制每个实验的柱状图
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# for idx, (experiment_name, experiment_data) in enumerate(data.items()):
#     ax = axes[idx]  # 获取当前图的轴
#     accuracy = experiment_data["GlobalClassAcc"]
#     method = experiment_data["model"]
#
#     # 计算标准差
#     mean_acc = np.mean(accuracy)
#     std_acc = np.std(accuracy)
#
#     # X轴是类别（假设有13个类别）
#     categories = np.arange(1, len(accuracy) + 1)
#
#     # 绘制柱状图
#     ax.bar(categories, accuracy, color=colors, width=0.7)
#
#     # 设置图表标题和标签，标题包括标准差信息
#     ax.text(0.5, -0.35, f"({chr(97 + idx)}){method}\n平均值: {mean_acc:.2f}, 标准差: {std_acc:.2f}",
#             fontsize=22, ha='center', va='center', transform=ax.transAxes)
#
#     # ax.set_title(f"{method}\n平均值: {mean_acc:.2f}, 标准差: {std_acc:.2f}")
#     # ax.set_title(f'{method} - 标准差: {std_dev:.2f}')  # 标题包括标准差
#     ax.set_xlabel('类别',fontsize=18)  # 设置X轴标签
#     ax.set_ylabel('准确率 (%)',fontsize=18)  # 设置Y轴标签
#     ax.set_xticks(categories)  # 设置X轴刻度为类别索引
#     ax.set_ylim(0, 100)  # 设置Y轴的范围为0到100（准确率百分比）
#
# # 调整布局
# plt.tight_layout()
# savepath = os.path.join("D:\\FedLearning\\毕设参考文献\\数据集","类结果分布327-song.pdf")
# plt.savefig(savepath, format='pdf')
# plt.show()
#









# import os
#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
# path ="D:\\FedLearning\\毕设参考文献\\结果保存\\混淆矩阵"
# # 图片路径和标题
# images = ["12Before.png", "12After.png", "14Before.png", "14After.png"]
# titles = ["客户端12-全局分类头对齐前的混淆矩阵", "客户端12-全局分类头对齐后的混淆矩阵", "客户端14-全局分类头对齐前的混淆矩阵", "客户端14-全局分类头对齐后的混淆矩阵"]
#
# # 创建 2x2 子图
# fig, axes = plt.subplots(2, 2, figsize=(18, 18))
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# for i, ax in enumerate(axes.flatten()):
#     imagePath = os.path.join(path, images[i])
#     img = mpimg.imread(imagePath)
#     ax.imshow(img)
#     # ax.text(0.5, -0.25, f"({chr(97 + i)}){titles[i]}",
#     #         fontsize=20, ha='center', va='center', transform=ax.transAxes)
#     ax.text(0.5, -0.05, f"({chr(97 + i)}) {titles[i]}",
#             fontsize=20, ha='center', va='center', transform=ax.transAxes)
#
#
#     ax.axis('off')  # 去掉坐标轴
#
# plt.tight_layout()
# plt.savefig("D:\\FedLearning\\毕设参考文献\\结果保存\\混淆矩阵\\combined_image.png", dpi=400)  # 保存为 PNG 文件
# plt.show()









#
# import pandas as pd
# from matplotlib import pyplot as plt
# import os
# import ujson
#
# # colorl = ["#2CA02C", "#D62728", "#1F77B4", "#ea894b", "#a95465"]
# colorl = ["#335c67", "#a98467", "#e09f3e", "#9e2a2b", "#540b0e","#3d5a80","#98c1d9","#724e91","#8e6c4d","#293241","#708d81","#ff9500","#fed9b7","#0466c8","#d62828"]
#
# # 创建一个新的图形
# plt.figure(figsize=(10, 6))
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# # 设置边框
# ax = plt.gca()  # 获取边框
# bwith = 3
# ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
# ax.spines['left'].set_linewidth(bwith)  # 图框左边
# ax.spines['top'].set_linewidth(bwith)  # 图框上边
# ax.spines['right'].set_linewidth(bwith)  # 图框右边
#
# # 取消边框
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['top'].set_visible(True)
# ax.spines['right'].set_visible(True)
#
# # 设置坐标轴和网格
# ax.tick_params(axis='both', labelsize=20)
# ax.tick_params(axis="both", which="major", direction="in", width=0.5, length=5, pad=5)
# ax.grid(linestyle="--", alpha=0.8, axis="both")
# # ax.set_xlabel('Iteration', x=0.5, y=-2, fontdict={'family': 'Times New Roman', 'size': 30}, labelpad=-2)
# # ax.set_ylabel('Accuracy(%)', fontdict={'family': 'Times New Roman', 'size': 28})
# ax.set_xlabel('通信轮次', fontsize=20,fontname='SimSun')
# ax.set_ylabel('准确率', fontsize=20,fontname='SimSun')
#
# # 设置边框宽度
# bwith = 2
# ax.spines['bottom'].set_linewidth(bwith)
# ax.spines['left'].set_linewidth(bwith)
# ax.spines['top'].set_linewidth(bwith)
# ax.spines['right'].set_linewidth(bwith)
#
# # 创建一个新的放大框（放大部分图）
# newax = plt.axes([0.5, 0.19, 0.2, 0.25])
# # newax.set_ylim(70, 90)
# newax.set_ylim(82, 86)
# newax.tick_params(axis='both', labelsize=12)
# newax.tick_params(axis="both", which="major", direction="in", width=0.5, length=5, pad=5)
# newax.grid(linestyle="--", alpha=0.8, axis="both")
# # newax.set_xticks([40, 45, 50])
# newax.set_xticks([5, 10, 15,20])
#
# newax.spines['bottom'].set_linewidth(bwith)
# newax.spines['left'].set_linewidth(bwith)
# newax.spines['top'].set_linewidth(bwith)
# newax.spines['right'].set_linewidth(bwith)
#
# # 设置绘制的算法名称及样式
# num_clients = 20
# # algorithms = ["FedAvg", "FedAMP", "FedPHP", "FedAPA"]
# # algorithms = ["FedAvg","FedPHP","FedAMP","APPLE","FedALA","pFedHN","pFedLA","FedPAC","DBE","FedProto","FedGH","FedAPA"]
# algorithms = ["FedAvg","FedProx","FedPHP","FedAMP","pFedHN","MOON","APPLE","FedGH","FedDistill","FedProto","FPL","DBE","FedTGP","FedLFP_exp","FedLFP"]
# # markers = ['o', 's', '^', '>', '<']
#
# communication_rounds = list(range(0, 50))  # 生成轮次列表
#
# # 绘制每个算法的曲线
# for index, algorithm in enumerate(algorithms):
#     # 读取客户端数据
#     # filename = "shoulianxing-"+algorithm + "-" + str(num_clients) + "-0.1.json"
#     # path = os.path.join("results", "huatu", filename)
#     filename ="213memoryNew-"+ algorithm + "-" + str(num_clients) + "-0.1.json"
#     path = os.path.join("D:\\FedLearning\\毕设参考文献\\结果保存\\notRealSame", filename)
#     with open(path, "r") as f:
#         result = ujson.load(f)
#         if algorithm == "FedAPA" or algorithm == "FedLFP"or algorithm == "FedLFP_exp" :
#             best_index = result["best_acc"].index(max(result["best_acc"]))
#             linestyle = '-'  # 实线
#             linewidth = 3  # 设置较粗的线条
#         # if algorithm == "FedLFP" :
#         #     best_acc = result["best_acc"]
#         #     sorted_acc = sorted(best_acc)
#         #     middle_value = sorted_acc[1]  # 选择中间的值
#         #     best_index = best_acc.index(middle_value)
#         else:
#             best_index = result["best_acc"].index(max(result["best_acc"]))
#             linestyle = '--'  # 虚线
#             linewidth = 2  # 设置较细的线条
#         acc_record = result["experiment"][best_index]["acc_record"]
#
#         # 将准确率数据转换为百分比
#         if algorithm=="pFedLA":
#               accuracy = [acc_record[str(round)] for round in communication_rounds]
#         else:
#             accuracy = [acc_record[str(round)] * 100 for round in communication_rounds]
#         # ax.plot(communication_rounds, accuracy, label=algorithm, marker=markers[index], color=colorl[index], clip_on=False)
#             ax.plot(communication_rounds, accuracy, label=algorithm, color=colorl[index], linestyle=linestyle, linewidth=linewidth, clip_on=False)
#             # newax.plot(range(41, 51), accuracy[-10:],  color=colorl[index], clip_on=True)
#             newax.plot(range(6, 21), accuracy[5:20], color=colorl[index], linestyle=linestyle, linewidth=linewidth, clip_on=True)
#
#             # ax.plot(communication_rounds, accuracy, label=algorithm, clip_on=False)
#             # newax.plot(range(41, 51), accuracy[-10:], clip_on=True)
#
# # 添加图例
# ax.legend()
#
# # 保存图像
# plt.savefig(f'D:\\FedLearning\\毕设参考文献\\结果保存\\notRealSame\\324-song.pdf', bbox_inches='tight')
#
# # 显示图像
# plt.show()

import pandas as pd
from matplotlib import pyplot as plt
import os
import ujson

# colorl = ["#2CA02C", "#D62728", "#1F77B4", "#ea894b", "#a95465"]
# colorl = ["#335c67", "#a98467", "#e09f3e","#3d5a80","#724e91","#8e6c4d","#0466c8","#540b0e","#a95465","#d62828"]
colorl = ["#335c67", "#e09f3e","#724e91","#8e6c4d","#0466c8","#540b0e","#a95465","#ea894b","#d62828"]

# 创建一个新的图形
plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 设置边框
ax = plt.gca()  # 获取边框
bwith = 3
ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
ax.spines['left'].set_linewidth(bwith)  # 图框左边
ax.spines['top'].set_linewidth(bwith)  # 图框上边
ax.spines['right'].set_linewidth(bwith)  # 图框右边

# 取消边框
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# 设置坐标轴和网格

ax.tick_params(axis='both', labelsize=20)

ax.tick_params(axis="both", which="major", direction="in", width=0.5, length=5, pad=5)
ax.grid(linestyle="--", alpha=0.8, axis="both")
# ax.set_xlabel('Iteration', x=0.5, y=-2, fontdict={'family': 'Times New Roman', 'size': 30}, labelpad=-2)
# ax.set_ylabel('Accuracy(%)', fontdict={'family': 'Times New Roman', 'size': 28})
ax.set_xlabel('通信轮次', fontsize=20,fontname='SimSun')
ax.set_ylabel('准确率', fontsize=20,fontname='SimSun')

ax.set_yticks(range(30, 91, 10))

# 设置边框宽度
bwith = 2
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

# 创建一个新的放大框（放大部分图）
newax = plt.axes([0.5, 0.19, 0.2, 0.25])
# newax.set_ylim(70, 90)
newax.set_ylim(75, 86)
newax.tick_params(axis='both', labelsize=12)
newax.tick_params(axis="both", which="major", direction="in", width=0.5, length=5, pad=5)
newax.grid(linestyle="--", alpha=0.8, axis="both")
# newax.set_xticks([40, 45, 50])
# newax.set_xticks([5, 10, 15,20])
newax.set_xticks([15, 20,25,30])
# newax.set_xticks([20, 25,30,35])

newax.spines['bottom'].set_linewidth(bwith)
newax.spines['left'].set_linewidth(bwith)
newax.spines['top'].set_linewidth(bwith)
newax.spines['right'].set_linewidth(bwith)

# 设置绘制的算法名称及样式
num_clients = 20
# algorithms = ["FedAvg", "FedAMP", "FedPHP", "FedAPA"]
# algorithms = ["FedAvg","FedPHP","FedAMP","APPLE","FedALA","pFedHN","pFedLA","FedPAC","DBE","FedProto","FedGH","FedAPA"]
# algorithms = ["LG_FedAvg","FedGen","FedGH","FedProto","FedDistill","FedTGP","FedMRL","FedSSA","FedWL"]
# algorithms = ['LG','FedGen','FedProto','FedGH', 'FedTGP','FedSSA','FedKDDC','pFedAFM','FedRAL','FedIADA']
algorithms = ['LG','FedProto','FedTGP','FedSSA','FedKDDC','pFedAFM','FedRAL','FedLFP','FedPSS']
# markers = ['o', 's', '^', '>', '<']

communication_rounds = list(range(0, 50))  # 生成轮次列表

# 绘制每个算法的曲线
for index, algorithm in enumerate(algorithms):
    # 读取客户端数据
    # filename = "shoulianxing-"+algorithm + "-" + str(num_clients) + "-0.1.json"
    # path = os.path.join("results", "huatu", filename)
    filename ="HtFE4-"+ algorithm + "-" + str(num_clients) + "-0.1.json"
    # path = os.path.join("G:\联邦\新建文件夹\图片\shoulian", filename)
    path = os.path.join(r"..\results\temp", filename)
    with open(path, "r") as f:
        result = ujson.load(f)
        if algorithm == "FedPSS":
            best_index = result["best_acc"].index(max(result["best_acc"]))
            linestyle = '-'  # 实线
            linewidth = 3  # 设置较粗的线条
        elif algorithm == "FedIADA" :
            best_acc = result["best_acc"]
            sorted_acc = sorted(best_acc)
            middle_value = sorted_acc[1]  # 选择中间的值
            best_index = best_acc.index(middle_value)
            # best_index = result["best_acc"].index(max(result["best_acc"]))
            linestyle = '-'  # 实线
            linewidth = 3  # 设置较粗的线条
        # if algorithm == "FedLFP" :
        #     best_acc = result["best_acc"]
        #     sorted_acc = sorted(best_acc)
        #     middle_value = sorted_acc[1]  # 选择中间的值
        #     best_index = best_acc.index(middle_value)
        else:
            best_index = result["best_acc"].index(min(result["best_acc"]))
            linestyle = '-'  # 虚线
            linewidth = 2  # 设置较细的线条
        acc_record = result["experiment"][best_index]["acc_record"]

        # 将准确率数据转换为百分比
        if algorithm=="pFedLA":
              accuracy = [acc_record[str(round)] for round in communication_rounds]
        else:
            accuracy = [acc_record[str(round)] * 100 for round in communication_rounds]
        # ax.plot(communication_rounds, accuracy, label=algorithm, marker=markers[index], color=colorl[index], clip_on=False)
            ax.plot(communication_rounds, accuracy, label=algorithm, color=colorl[index], linestyle=linestyle, linewidth=linewidth, clip_on=False)
            # newax.plot(range(41, 51), accuracy[-10:],  color=colorl[index], clip_on=True)
            # newax.plot(range(6, 21), accuracy[5:20], color=colorl[index], linestyle=linestyle, linewidth=linewidth, clip_on=True)
            newax.plot(range(16, 31), accuracy[15:30], color=colorl[index], linestyle=linestyle, linewidth=linewidth, clip_on=True)
            # newax.plot(range(21, 36), accuracy[20:35], color=colorl[index], linestyle=linestyle, linewidth=linewidth, clip_on=True)

            # ax.plot(communication_rounds, accuracy, label=algorithm, clip_on=False)
            # newax.plot(range(41, 51), accuracy[-10:], clip_on=True)

# 添加图例
ax.legend()

# 保存图像
plt.savefig(f'G:\联邦\新建文件夹\图片\shoulian\\FedPSSshoulian.pdf', bbox_inches='tight')

# 显示图像
plt.show()
