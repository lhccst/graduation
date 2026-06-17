import os

import numpy as np

from dataset.data_utils import load_data
from mem_utils import MemReporter
import torch
from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class ServerBase(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.global_rounds = args.server["global_rounds"]  # 全局通信次数
        self.join_clients = None  # 某轮参加训练的用户

        self.num_clients = args.client["num_clients"]  # 用户数
        self.num_classes = args.num_classes  # 类别数
        self.distribution = args.train_config["distribution"]  # 用户数据分布

        self.clients = []  # 所有用户列表
        try:
            self.lowest_join_rate = args.server["lowest_join_rate"]  # 最低参与率
        except:
            self.lowest_join_rate = 1  # 如果没有设置最低参与率，则默认为 1
        self.reporter = MemReporter()
        self.maxAcc = 0
        self.global_class_acc = [0] * self.num_classes
        self.client_acc = [0] * self.num_clients
        self.save=False

    def initialize_clients(self, ClientClass):
        train_subsets, test_subsets = load_data(self.args.paths["train"], self.args.paths["test"])
        for client_id in range(self.num_clients):
            client = ClientClass(args=self.args,
                                 client_id=client_id,
                                 trainset=train_subsets[client_id],
                                 testset=test_subsets[client_id],
                                 statistic=self.args.statistic[client_id])
            self.clients.append(client)

    def select_join_clients(self):
        minimal_join_clients = self.num_clients * self.lowest_join_rate
        num_join_clients = np.random.randint(low=minimal_join_clients, high=self.num_clients + 1)
        selected_clients = np.random.choice(self.clients,
                                            size=num_join_clients,
                                            replace=False).tolist()
        return selected_clients

    # def evaluate(self):
    #     total_samples = 0
    #     total_correct = 0
    #     for client in self.clients:
    #         num_correct, num_samples = client.evaluate_test()
    #         total_samples += num_samples
    #         total_correct += num_correct
    #         print('client:',client.client_id,'的准确率：',(100* num_correct/num_samples))
    #     acc = total_correct / total_samples
    #     return acc

    def evaluate(self,global_epoch):

        total_samples = 0
        total_correct = 0
        global_class_correct = [0] * self.num_classes
        global_class_total = [0] * self.num_classes
        global_class_acc = [0] * self.num_classes
        client_acc = [0] * self.num_clients
        for client in self.clients:
            num_correct, num_samples, class_correct,class_total = client.evaluate_test()
            total_samples += num_samples
            total_correct += num_correct
            print('client:',client.client_id,'的准确率：',(100* num_correct/num_samples))
            client_acc[client.client_id] = (100* num_correct/num_samples)
            # 汇总每个类别的统计信息
            for i in range(self.num_classes):
                global_class_correct[i] += class_correct[i]
                global_class_total[i] += class_total[i]

        acc = total_correct / total_samples
        # 打印每个类别的全局准确率
        for i in range(self.num_classes):
            if global_class_total[i] > 0:
                global_class_acc[i] = 100 * global_class_correct[i] / global_class_total[i]
                print(f'Class {i} 的全局准确率：{global_class_acc[i] :.2f}%, 正确数：{global_class_correct[i]},样本数：{global_class_total[i]}')
            else:
                print(f'Class {i} 的全局准确率：N/A (no samples)')


        if acc > self.maxAcc:
            self.maxAcc = acc
            self.global_class_acc = global_class_acc
            self.client_acc =client_acc
            self.bestEpoch=global_epoch
            if(self.save):
                for clientSave in self.clients:
                    best_model_path= "/root/autodl-tmp/FedPSY226/results/kronoDroid/"+str(self.args.algorithm)+"-models/"+str(global_epoch)
                    model_path = best_model_path+"/"+str(clientSave.client_id)+".pth"
                    if not os.path.exists(best_model_path):
                        os.makedirs(best_model_path)
                    torch.save({
                        "epoch": global_epoch,
                        "model_state_dict": clientSave.model.state_dict(),
                    }, model_path)


        return acc

    def getReport(self):
        uploadCom,downloadCom=self.reporter.print_communication_stats()
        self.reporter.report()
        usedMemory =self.reporter.total_used_memory
        return (uploadCom,downloadCom,usedMemory)

    def drawFeature(self,bestEpoch):
        global_test_features = []
        global_test_labels = []
        for client in self.clients:
            # client.draw(bestEpoch=27)
            test_features, test_labels = client.extractFeature(bestEpoch=bestEpoch)
            global_test_features.append(test_features)
            global_test_labels.append(test_labels)
        # 使用 t-SNE 降维

        global_test_features = np.vstack(global_test_features)
        global_test_labels = np.concatenate(global_test_labels)

        # tsne = TSNE(n_components=2, random_state=42)
        # features_2d = tsne.fit_transform(global_test_features)
        #
        # # 绘制特征分布图
        # plt.figure(figsize=(8, 8))
        # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=global_test_labels, cmap='tab10', s=50, alpha=0.7)
        # plt.colorbar(scatter)
        # plt.title("Feature Distribution (t-SNE)")
        # plt.xlabel("Component 1")
        # plt.ylabel("Component 2")
        # plt.show()
        # 使用 t-SNE 进行 3D 降维

        # 选择每个类别最多 200 个样本
        selected_features = []
        selected_labels = []
        unique_labels = np.unique(global_test_labels)  # 获取所有类别

        for label in unique_labels:
            indices = np.where(global_test_labels == label)[0]  # 获取该类别的索引
            selected_indices = indices[:5000]  # 只取前 200 个
            selected_features.append(global_test_features[selected_indices])
            selected_labels.append(global_test_labels[selected_indices])

        # 合并筛选后的数据
        selected_features = np.vstack(selected_features)
        selected_labels = np.concatenate(selected_labels)

        # tsne = TSNE(n_components=3, random_state=42)
        # features_3d = tsne.fit_transform(selected_features)
        #
        # # 绘制 3D 散点图
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')  # 设置 3D 坐标系
        #
        # scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
        #                      c=selected_labels, cmap='tab10', s=50, alpha=0.7)
        #
        # ax.set_title("3D Feature Distribution (t-SNE)")
        # ax.set_xlabel("Component 1")
        # ax.set_ylabel("Component 2")
        # ax.set_zlabel("Component 3")
        # fig.colorbar(scatter)
        #
        # plt.show()

        tsne = TSNE(n_components=2, random_state=42, perplexity=10, learning_rate=50)
        features_2d = tsne.fit_transform(selected_features)

        # 绘制特征分布图
        # plt.figure(figsize=(8, 8))
        # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=selected_labels, cmap='tab20', s=50, alpha=0.7)
        # plt.colorbar(scatter)
        # plt.title("Feature Distribution (t-SNE)")
        # plt.xlabel("Component 1")
        # plt.ylabel("Component 2")
        # plt.show()
        # 定义每个类别的颜色
        # custom_colors = [
        #     'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink',
        #     'yellow', 'brown', 'gray', 'lime', 'teal', 'magenta'
        # ]
        custom_colors = ['#dab49d', '#adc178', '#93b7be', '#457b9d', '#0a9396', '#ee9b00', '#e9c46a',
                  '#fee440', '#f7e1d7', '#f07167', '#6d6875', '#bcb8b1', '#a49fd5']

        # 将标签映射到颜色
        colors = [custom_colors[label] for label in selected_labels]

        # 绘制特征分布图
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, s=50, alpha=0.7)
        if self.args.algorithm=="FedCPCL":
            name= "FedLFP"
        else:
            name=str(self.args.algorithm)
        plt.title(name+"-Feature Distribution (t-SNE)",fontsize=20)
        plt.xlabel("1st dimension",fontsize=20)
        plt.ylabel("2st dimension",fontsize=20)

        # 添加图例
        for i, color in enumerate(custom_colors):
            plt.scatter([], [], c=color, label=f"Class {i}", s=50)
        plt.legend(loc="upper right", fontsize=10)
        savepath= "/root/autodl-tmp/FedPSY226/results/kronoDroid/" + name + ".pdf"
        plt.savefig(savepath, format='pdf')
        plt.show()
