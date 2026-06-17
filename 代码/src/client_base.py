import copy

import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from models.models import BaseHeadSplit

class ClientBase:
    def __init__(self, args, client_id, trainset, testset, statistic):
        self.args = args
        self.algorithm = args.algorithm  # 算法名称
        self.device = args.device
        self.flType= args.flType

        self.client_id = client_id  # 用户 id
        self.num_clients = args.client["num_clients"]  # 用户数
        self.num_classes = args.num_classes  # 类别数
        self.trainset = trainset  # 训练集
        self.testset = testset  # 测试集
        self.sample_size = len(self.trainset)  # 训练集样本数
        self.statistic = statistic  # 各类标签的样本量统计

        self.batch_size = args.client["batch_size"]  # 本地训练批次大小
        self.local_epochs = args.client["local_epochs"]  # 本地训练轮数
        self.learning_rate = args.client["learning_rate"]  # 本地学习率
        self.loss = nn.CrossEntropyLoss()  # 损失函数
        if self.flType!="HtFL":
            self.model = copy.deepcopy(args.model)
        if self.flType=="HtFL":
            self.model = BaseHeadSplit(args, self.client_id).to(self.device)

    def get_train_loader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def get_test_loader(self):
        # print(f"Number of samples in test set: {len(self.testset)}")
        return DataLoader(self.testset, batch_size=self.batch_size, drop_last=False, shuffle=True)

    # def evaluate_test(self):
    #     test_loader = self.get_test_loader()
    #
    #
    #     self.model.eval()
    #     total_correct = 0
    #     total_samples = 0
    #     with torch.inference_mode():
    #         for x, y in test_loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             output = self.model(x)
    #             total_correct += torch.sum(torch.argmax(output, dim=1) == y).item()
    #             total_samples += len(y)
    #     return total_correct, total_samples


    def evaluate_test(self):
        test_loader = self.get_test_loader()
        self.model.eval()

        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        with torch.inference_mode():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                preds = torch.argmax(output, dim=1)

                # 统计每个类别的正确预测数和总样本数
                for i in range(self.num_classes):
                    class_mask = (y == i)
                    class_total[i] += torch.sum(class_mask).item()
                    class_correct[i] += torch.sum((preds == y) & class_mask).item()

        # 计算并打印每个类别的准确率
        # for i in range(self.num_classes):
        #     if class_total[i] > 0:
        #         accuracy = 100 * class_correct[i] / class_total[i]
        #         print(f'Class {i} Accuracy: {accuracy:.2f}%')
        #     else:
        #         print(f'Class {i} Accuracy: N/A (no samples)')

        # 返回总的正确预测数和总样本数
        total_correct = sum(class_correct)
        total_samples = sum(class_total)
        return total_correct, total_samples,class_correct,class_total

    def extractFeature(self,bestEpoch):
        best_model_path = "/root/autodl-tmp/fedwl/results/kronoDroid/" + str(self.args.algorithm) + "-models/" + str(bestEpoch)
        model_path = best_model_path + "/" + str(self.client_id) + ".pth"
        bestmodel = self.model

        checkpoint = torch.load(model_path,map_location=self.device)
        bestmodel.load_state_dict(checkpoint["model_state_dict"])
        bestmodel.eval()  # 进入评估模式
        test_loader = self.get_test_loader()

        features, labels = [], []

        with torch.no_grad():  # 关闭梯度计算，加快推理速度
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device),targets.to(self.device)
                outputs = bestmodel.base(inputs)  # 直接前向传播
                features.append(outputs.cpu().numpy())  # 转换为 numpy
                labels.extend(targets.cpu().numpy())  # 保存标签

            test_features=np.vstack(features)
            test_labels=np.array(labels)

        return test_features,test_labels