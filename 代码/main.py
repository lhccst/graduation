import argparse
import copy
import math
import time

import numpy as np
import torch
import yaml
from torch import nn

from mem_utils import MemReporter
from models.FedAvgCNN import FedAvgCNN
from dataset.generate_dataset import generate_LTdataset
from dataset.generate_dataset import generate_dataset

from src.APA_ALA.server_APA_ALA import ServerAPA_ALA
from src.APA_CCPL.server_APA_CCPL import ServerAPA_CCPL
from src.APA_CPL.server_APA_CPL import ServerAPA_CPL
from src.APA_PHP.server_APA_PHP import ServerAPA_PHP
from src.APPLE.server_APPLE import ServerAPPLE
from src.DBE.server_DBE import ServerDBE
from src.FedALA.server_FedALA import ServerFedALA
from src.FedAMP.server_FedAMP import ServerFedAMP
from src.FedAPA.server_FedAPA import ServerFedAPA
from src.FedAvg.server_FedAvg import ServerFedAvg
from src.FedRAL.server_FedRAL import ServerFedRAL
from src.FedSC.server_FedSC import ServerFedSC
from src.LG.server_LG import ServerLG
from src.FedCPCL.server_FedCPCL import ServerFedCPCL
from src.FedDistill.server_FedDistill import ServerFedDistill
from src.FedGH.server_FedGH import ServerFedGH
from src.FedPAA.server_FedPAA import ServerFedPAA
from src.FedPAA_GH.server_FedPAA_GH import ServerFedPAA_GH
from src.FedPCL.server_FedPCL import ServerFedPCL
from src.FedPHP.server_FedPHP import ServerFedPHP
from src.FedProto.server_FedProto import ServerFedProto
from src.MOON.server_MOON import ServerMOON
from models.models import Net, BaseHeadMerge
from src.pFedHN.server_pFedHN import ServerFedHN, generate_server_model
from utils import save_result
from models.ResNet import ResNet18
from models.ResNet_T import resnet4
# from src.pFedLA import server_pFedLA
from src.FPL.server_FPL import ServerFPL
from src.FedProx.server_FedProx import ServerFedProx
from src.FedTGP.server_FedTGP import ServerFedTGP
from src.FedAPAnoSplit.server_FedAPAnoSplit import ServerFedAPAnoSplit
from utils import readable_size
from src.FedREP.server_FedREP import ServerFedREP
from src.FedPAC.server_FedPAC import ServerFedPAC
from src.FedAPARep.server_FedAPARep import ServerFedAPARep
from src.FedRepAPA.server_FedRepAPA import ServerFedRepAPA
from src.FedASAPA.server_FedASAPA import ServerFedASAPA
from src.FedSSA.server_FedSSA import ServerFedSSA
from src.FedPSYSSA.server_FedPSYSSA import ServerFedPSYSSA
from src.FedPSYSSALong.server_FedPSYSSALong import ServerFedPSYSSALong
from src.FedGen.server_FedGen import ServerFedGen
from src.FedMRL.server_FedMRL import ServerFedMRL
from src.pFedAFM.server_pFedAFM import ServerpFedAFM
# def prepare_models(args):
#
#     if args.dataset == "mnist":
#         args.num_classes = 10  # 类别数
#         dim = 16 * 4 * 4  # 具体数值需要根据样本尺寸变化，mnist 图片尺寸为 1*28*28，黑白
#
#         args.model = Net(in_channels=1, num_classes=10, dim=dim)  # 基础分类模型，对这个模型后续训练过程不能有任何改动
#         args.feature_dim = 84
#
#     elif args.dataset == "fmnist":
#         args.num_classes = 10  # 类别数
#         dim = 16 * 4 * 4  # fmnist 图片尺寸为 1*28*28，黑白
#
#         args.model = Net(in_channels=1, num_classes=10, dim=dim)
#         args.feature_dim = 84
#
#     elif args.dataset == "cifar10":
#         args.num_classes = 10  # 类别数
#         dim = 16 * 5 * 5  # cifar10 图片尺寸为 3*32*32，彩色
#
#         args.model = Net(in_channels=3, num_classes=10, dim=dim)
#         args.feature_dim = 84
#
#     elif args.dataset == "cifar100":
#         args.num_classes = 100  # 类别数
#         dim = 16 * 5 * 5  # cifar100 图片尺寸为 3*32*32，彩色
#
#         args.model = Net(in_channels=3, num_classes=100, dim=dim)
#         args.feature_dim = 84
#
#     elif args.dataset == "kronoDroid":
#         args.num_classes = 13  # 类别数
#         dim = 16 * 1 * 1  # cifar100 图片尺寸为 3*18*18，彩色
#         if(args.model=="ResNet18"):
#             args.model = ResNet18()
#             args.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#             args.model.fc = torch.nn.Linear(512, args.num_classes)  # 将最后的全连
#         else:
#             args.model = Net(in_channels=3, num_classes=13, dim=dim)
#         args.feature_dim = 84
#
#     else:
#         raise NotImplementedError
#
#     if args.algorithm == "pFedHN":
#         args.server_model = generate_server_model(args, dim)
#
#     return
#

def prepare_modelsDIM(args):
    if args.dataset == "fmnist":
        args.num_classes = 10  # 类别数
        args.in_channels=1
        args.FedAvgCNNDim = 1024
        args.feature_dim = 84

    elif args.dataset == "cifar10":
        args.num_classes = 10  # 类别数
        args.in_channels=3
        args.feature_dim = 84
        args.FedAvgCNNDim = 1600

    elif args.dataset == "cifar100":
        args.num_classes = 100  # 类别数
        args.in_channels=3
        args.feature_dim = 84
        args.FedAvgCNNDim = 1600

    elif args.dataset == "kronoDroid":
        args.num_classes = 13  # 类别数
        args.in_channels=3
        args.feature_dim = 84
        args.FedAvgCNNDim = 64

    else:
        raise NotImplementedError
    return



def prepare_models(args):
    dim=0
    in_channels=0
    if args.dataset == "mnist":
        args.num_classes = 10  # 类别数
        args.feature_dim = 84
        dim = 16 * 4 * 4  # 具体数值需要根据样本尺寸变化，mnist 图片尺寸为 1*28*28，黑白
        in_channels=1

    elif args.dataset == "fmnist":
        args.num_classes = 10  # 类别数
        dim = 16 * 4 * 4  # fmnist 图片尺寸为 1*28*28，黑白
        args.in_channels=1
        # args.model = Net(in_channels=1, num_classes=10, dim=dim)
        args.FedAvgCNNDim = 1024
        args.feature_dim = 84

    elif args.dataset == "cifar10":
        args.num_classes = 10  # 类别数
        dim = 16 * 5 * 5  # cifar10 图片尺寸为 3*32*32，彩色
        args.in_channels=3
        # args.model = Net(in_channels=3, num_classes=10, dim=dim)
        args.feature_dim = 84
        args.FedAvgCNNDim = 1600

    elif args.dataset == "cifar100":
        args.num_classes = 100  # 类别数
        dim = 16 * 5 * 5  # cifar100 图片尺寸为 3*32*32，彩色
        args.in_channels=3
        # args.model = Net(in_channels=3, num_classes=100, dim=dim)
        args.feature_dim = 84
        args.FedAvgCNNDim = 1600

    elif args.dataset == "kronoDroid":
        args.num_classes = 13  # 类别数
        dim = 16 * 1 * 1  # cifar100 图片尺寸为 3*18*18，彩色
        args.in_channels=3
        args.feature_dim = 84
        args.FedAvgCNNDim = 64

    else:
        raise NotImplementedError

    if args.algorithm == "pFedHN":
        args.server_model = generate_server_model(args, dim)

    if args.modelName =='LeNet5':
        args.model = Net(in_channels=args.in_channels, num_classes=args.num_classes, dim=dim)  # 基础分类模型，对这个模型后续训练过程不能有任何改动
    elif args.modelName=="ResNet18":
        args.model = ResNet18()
        args.model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        args.model.fc = torch.nn.Linear(512, args.num_classes)  # 将最后的全连
    elif args.modelName=="resnet4":
        args.model = resnet4(num_classes=args.num_classes)
    elif args.modelName=="FedAvgCNN":
        args.model = FedAvgCNN(in_features=args.in_channels, num_classes=args.num_classes, dim=args.FedAvgCNNDim).to(args.device)

    print(f"Model Framework: {args.model}")
    return













def run(args):
    if args.flType == "HtFL":
        results = {"HtFL-Models": [], "experiment": [], "best_acc": [], "upload": [], "download": [], "allTime": [],
               "iterationTime": [], "memory": []}
    else:
        results = {"experiment": [], "best_acc": [], "upload": [], "download": [], "allTime": [], "iterationTime": [],"memory":[]}
    for repeat_time in range(args.num_repeat_times):
        print(f"================= Repeat Time: {repeat_time + 1} =================")
        # generate_LTdataset(args)
        args.train_config, args.statistic = generate_dataset(args)
        if args.flType=="HtFL":
            prepare_modelsDIM(args)
            # Generate args.models
            if args.model_family == "HtFE2":
                args.models = [
                    'FedAvgCNN(in_features=args.in_channels, num_classes=args.num_classes, dim=args.FedAvgCNNDim)',
                    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                    # 'resnet4(num_classes=args.num_classes)',
                ]

            elif args.model_family == "HtFE3":
                args.models = [
                    'resnet10(num_classes=args.num_classes)',
                    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                ]

            elif args.model_family == "HtFE4":
                args.models = [
                    'FedAvgCNN(in_features=args.in_channels,num_classes=args.num_classes, dim=args.FedAvgCNNDim)',
                    'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                    'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
                ]

            elif args.model_family == "HtFE5":
                args.models = [
                    'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                    'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                    # 'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
                    'FedAvgCNN(in_features=args.in_channels, num_classes=args.num_classes, dim=args.FedAvgCNNDim)'
                ]

            elif args.model_family == "HtFE8":
                args.models = [
                    'FedAvgCNN(in_features=args.in_channels, num_classes=args.num_classes, dim=args.FedAvgCNNDim)',
                    # 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)',
                    'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                    'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
                ]

            elif args.model_family == "HtFE9":
                args.models = [
                    'resnet4(num_classes=args.num_classes)',
                    'resnet6(num_classes=args.num_classes)',
                    'resnet8(num_classes=args.num_classes)',
                    'resnet10(num_classes=args.num_classes)',
                    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
                ]

            elif args.model_family == "HtFE8-HtC4":
                args.models = [
                    'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.FedAvgCNNDim)',
                    'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                    'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
                ]
                args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'
                args.heads = [
                    'Head(hidden_dims=[512], num_classes=args.num_classes)',
                    'Head(hidden_dims=[512, 512], num_classes=args.num_classes)',
                    'Head(hidden_dims=[512, 256], num_classes=args.num_classes)',
                    'Head(hidden_dims=[512, 128], num_classes=args.num_classes)',
                ]

            elif args.model_family == "Res34-HtC4":
                args.models = [
                    'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                ]
                args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'
                args.heads = [
                    'Head(hidden_dims=[512], num_classes=args.num_classes)',
                    'Head(hidden_dims=[512, 512], num_classes=args.num_classes)',
                    'Head(hidden_dims=[512, 256], num_classes=args.num_classes)',
                    'Head(hidden_dims=[512, 128], num_classes=args.num_classes)',
                ]

            elif args.model_family == "HCNNs8":
                args.models = [
                    'CNN(num_cov=1, hidden_dims=[], in_features=1, num_classes=args.num_classes)',
                    'CNN(num_cov=2, hidden_dims=[], in_features=1, num_classes=args.num_classes)',
                    'CNN(num_cov=1, hidden_dims=[512], in_features=1, num_classes=args.num_classes)',
                    'CNN(num_cov=2, hidden_dims=[512], in_features=1, num_classes=args.num_classes)',
                    'CNN(num_cov=1, hidden_dims=[1024], in_features=1, num_classes=args.num_classes)',
                    'CNN(num_cov=2, hidden_dims=[1024], in_features=1, num_classes=args.num_classes)',
                    'CNN(num_cov=1, hidden_dims=[1024, 512], in_features=1, num_classes=args.num_classes)',
                    'CNN(num_cov=2, hidden_dims=[1024, 512], in_features=1, num_classes=args.num_classes)',
                ]

            elif args.model_family == "ViTs":
                args.models = [
                    'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)',
                    'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)',
                    'torchvision.models.vit_l_16(image_size=32, num_classes=args.num_classes)',
                    'torchvision.models.vit_l_32(image_size=32, num_classes=args.num_classes)',
                ]

            elif args.model_family == "HtM10":
                args.models = [
                    'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
                    'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
                    'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
                    'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)',
                    'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)'
                ]

            elif args.model_family == "NLP_all":
                args.models = [
                    'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                    'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                    'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)',
                    'TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, num_classes=args.num_classes)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)'
                ]

            elif args.model_family == "NLP_Transformers-nhead=8":
                args.models = [
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
                ]

            elif args.model_family == "NLP_Transformers-nlayers=4":
                args.models = [
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                ]

            elif args.model_family == "NLP_Transformers":
                args.models = [
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
                    'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
                ]

            elif args.model_family == "MLPs":
                args.models = [
                    'AmazonMLP(feature_dim=[])',
                    'AmazonMLP(feature_dim=[200])',
                    'AmazonMLP(feature_dim=[500])',
                    'AmazonMLP(feature_dim=[1000, 500])',
                    'AmazonMLP(feature_dim=[1000, 500, 200])',
                ]

            elif args.model_family == "MLP_1layer":
                args.models = [
                    'AmazonMLP(feature_dim=[200])',
                    'AmazonMLP(feature_dim=[500])',
                ]

            elif args.model_family == "MLP_layers":
                args.models = [
                    'AmazonMLP(feature_dim=[500])',
                    'AmazonMLP(feature_dim=[1000, 500])',
                    'AmazonMLP(feature_dim=[1000, 500, 200])',
                ]

            else:
                raise NotImplementedError
            for model in args.models:
                print(model)
            results["HtFL-Models"]= args.models

            if args.algorithm == "LG":
                args.head = nn.Linear(args.feature_dim, args.num_classes)
                server = ServerLG(args)

            elif args.algorithm == "FedGen":
                args.head = nn.Linear(args.feature_dim, args.num_classes)
                args.noise_dim=512
                args.hidden_dim=512
                server = ServerFedGen(args)

            elif args.algorithm == "FedMRL":
                # args.head = nn.Linear(args.feature_dim, args.num_classes)
                args.sub_feature_dim = args.feature_dim
                server = ServerFedMRL(args)


            elif args.algorithm == "MOON":
                server = ServerMOON(args)

            elif args.algorithm == "FedPHP": #对比实验 FedPHP: Federated Personalization with Inherited Private Models
                server = ServerFedPHP(args)

            elif args.algorithm == "FedAMP": #对比实验 Personalized Cross-Silo Federated Learning on Non-IID Data
                server = ServerFedAMP(args)

            elif args.algorithm == "APPLE":
                server = ServerAPPLE(args)

            elif args.algorithm == "FedDistill": #对比实验 FedDistill: Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning
                server = ServerFedDistill(args)

            elif args.algorithm == "FedProto": #对比实验 FedProto: Federated Prototype Learning across Heterogeneous Clients
                server = ServerFedProto(args)


            elif args.algorithm == "FedGH":  #FedGH: Heterogeneous Federated Learning with Generalized Global Header
                args.head = nn.Linear(args.feature_dim, args.num_classes)
                server = ServerFedGH(args)


            elif args.algorithm == "FedCPCL":  # Federated Contrastive Prototype Centroid Learning
                server = ServerFedCPCL(args)


            elif args.algorithm == "FPL":
                server = ServerFPL(args)

            elif args.algorithm == "FedTGP":
                server = ServerFedTGP(args)

            elif args.algorithm == "FedProx":
                server = ServerFedProx(args)

            elif args.algorithm == "FedREP":
                server = ServerFedREP(args)

            elif args.algorithm == "FedPAC":
                server = ServerFedPAC(args)

            elif args.algorithm == "FedAPARep":
                server = ServerFedAPARep(args)

            elif args.algorithm == "FedRepAPA":
                server = ServerFedRepAPA(args)

            elif args.algorithm == "FedASAPA":  # Federated Adaptive Parameter Aggregation
                server = ServerFedASAPA(args)

            elif args.algorithm == "FedSSA":  # Federated Adaptive Parameter Aggregation
                args.head = nn.Linear(args.feature_dim, args.num_classes)
                # args.head = nn.Sequential(
                #     nn.Linear(args.feature_dim, 512),  # 第一层 Linear，输入特征维度到 512
                #     nn.ReLU(),  # 激活函数
                #     nn.Linear(512, args.num_classes)  # 第二层 Linear，从 512 到类别数
                # )

                server = ServerFedSSA(args)

            elif args.algorithm == "FedPSYSSA":  # Federated Adaptive Parameter Aggregation
                args.head = nn.Linear(args.feature_dim, args.num_classes)
                args.projection = nn.Linear(args.feature_dim, args.feature_dim)
                # args.projection = nn.Sequential(
                #     nn.Linear(args.feature_dim, 256),  # 第一层 Linear，输入特征维度到 512
                #     nn.ReLU(),  # 激活函数
                #     nn.Linear(256, args.feature_dim)  # 第二层 Linear，从 512 到类别数
                # )
                # args.head = nn.Sequential(
                #     nn.Linear(args.feature_dim, args.feature_dim),  # 第一层 Linear，输入特征维度到 512
                #     nn.ReLU(),  # 激活函数
                #     nn.Linear(args.feature_dim, args.num_classes)  # 第二层 Linear，从 512 到类别数
                # )
                server = ServerFedPSYSSA(args)


            elif args.algorithm == "FedPSYSSALong":  # Federated Adaptive Parameter Aggregation
                args.head = nn.Linear(args.feature_dim, args.num_classes)
                args.projection = nn.Linear(args.feature_dim, args.feature_dim)
                server = ServerFedPSYSSALong(args)

            elif args.algorithm == "pFedAFM":  # Federated Adaptive Parameter Aggregation
                args.head = nn.Linear(args.feature_dim, args.num_classes)
                server = ServerpFedAFM(args)

            elif args.algorithm == "FedRAL":
                args.head = nn.Linear(args.feature_dim, args.num_classes)
                server = ServerFedRAL(args)

            elif args.algorithm == "FedSC":
                server = ServerFedSC(args)
            else:
                raise NotImplementedError
        else:
            results["model"] = args.modelName
            prepare_models(args)
            if args.algorithm == "FedAvg":
                args.model = args.model.to(args.device)
                server = ServerFedAvg(args)

            elif args.algorithm == "MOON":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerMOON(args)

            elif args.algorithm == "FedPHP": #对比实验 FedPHP: Federated Personalization with Inherited Private Models
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedPHP(args)

            elif args.algorithm == "FedAMP": #对比实验 Personalized Cross-Silo Federated Learning on Non-IID Data
                args.model = args.model.to(args.device)
                server = ServerFedAMP(args)

            elif args.algorithm == "APPLE":
                args.model = args.model.to(args.device)
                server = ServerAPPLE(args)

            elif args.algorithm == "FedDistill": #对比实验 FedDistill: Global Model Distillation for Local Model De-Biasing in Non-IID Federated Learning
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                # args.model = args.model.to(args.device)
                server = ServerFedDistill(args)

            elif args.algorithm == "FedProto": #对比实验 FedProto: Federated Prototype Learning across Heterogeneous Clients
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedProto(args)

            elif args.algorithm == "FedPCL":  #对比实验 FedPCL: Learning to Blend Representations for Federated Prototype Learning
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedPCL(args)

            elif args.algorithm == "pFedHN":  #对比试验 Personalized Federated Learning using Hypernetworks
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)

                args.server_model = args.server_model.to(args.device)

                server = ServerFedHN(args)

            elif args.algorithm == "DBE":  #Eliminating Domain Bias for Federated Learning in Representation Space
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerDBE(args)

            elif args.algorithm == "FedGH":  #FedGH: Heterogeneous Federated Learning with Generalized Global Header
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedGH(args)

            elif args.algorithm == "FedALA": #对比试验 FedALA: Adaptive Local Aggregation for Personalized Federated Learning
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedALA(args)

            elif args.algorithm == "FedCPCL":  # Federated Contrastive Prototype Centroid Learning
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedCPCL(args)

            elif args.algorithm == "FedPAA":  # Federated Prototype-based Affinity Aggregation
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedPAA(args)

            elif args.algorithm == "FedPAA_GH":  # Federated Prototype-based Affinity Aggregation with Global Head
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedPAA_GH(args)

            elif args.algorithm == "FedAPA":  # Federated Adaptive Parameter Aggregation
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedAPA(args)

            elif args.algorithm == "FedAPAnoSplit":  # Federated Adaptive Parameter Aggregation
                args.model = args.model.to(args.device)
                server = ServerFedAPAnoSplit(args)

            elif args.algorithm == "APA_CPL":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerAPA_CPL(args)

            elif args.algorithm == "APA_ALA":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerAPA_ALA(args)

            elif args.algorithm == "APA_PHP":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerAPA_PHP(args)

            elif args.algorithm == "APA_CCPL":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerAPA_CCPL(args)

            # elif args.algorithm == "pFedLA":
            #     head = copy.deepcopy(args.model.fc)
            #     args.model.fc = nn.Identity()
            #     args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
            #     server = server_pFedLA(args)

            elif args.algorithm == "FPL":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFPL(args)

            elif args.algorithm == "FedTGP":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedTGP(args)

            elif args.algorithm == "FedProx":
                args.model = args.model.to(args.device)
                server = ServerFedProx(args)
                # head = copy.deepcopy(args.model.fc)
                # args.model.fc = nn.Identity()
                # args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                # server = ServerFedProx(args)

            elif args.algorithm == "FedREP":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedREP(args)

            elif args.algorithm == "FedPAC":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedPAC(args)

            elif args.algorithm == "FedAPARep":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedAPARep(args)

            elif args.algorithm == "FedRepAPA":
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedRepAPA(args)



            elif args.algorithm == "FedASAPA":  # Federated Adaptive Parameter Aggregation
                head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedASAPA(args)

            elif args.algorithm == "FedPSYSSA":
                head = copy.deepcopy(args.model.fc)
                args.projection = nn.Linear(args.feature_dim, args.feature_dim)
                args.head = head
                args.model.fc = nn.Identity()
                args.model = BaseHeadMerge(base=args.model, head=head).to(args.device)
                server = ServerFedPSYSSA(args)

            else:
                raise NotImplementedError

        all_start_time = time.time()
        draw=True
        if(draw):
            if(args.algorithm == "FedCPCL"):
                bestEpoch=37
            if (args.algorithm == "FedSC"):
                bestEpoch = 42
            if(args.algorithm == "FedTGP"):
                bestEpoch= 20
            if(args.algorithm == "FedProto"):
                bestEpoch= 41
            if (args.algorithm == "FedDistill"):
                bestEpoch = 30

            server.drawFeature(bestEpoch=bestEpoch)
            return
        acc_record, uploadCom, downloadCom,memoryCom = server.train()
        all_end_time = time.time()
        time_all = all_end_time - all_start_time

        # result = {"statistic": args.statistic, "acc_record": acc_record, "GlobalClassAcc":server.global_class_acc,"ClientAcc":server.client_acc}
        result = {"statistic": args.statistic, "acc_record": acc_record}

        results["experiment"].append(result)
        results["best_acc"].append(max(acc_record.values()))
        results["upload"].append(uploadCom)
        results["download"].append(downloadCom)
        results["memory"].append(memoryCom)
        results["allTime"].append(time_all)
        results["iterationTime"].append(time_all / args.server['global_rounds'])
        # results["GlobalClassAcc"].append(server.global_class_acc)
        # results["ClientAcc"].append(server.client_acc)

    print("============ Saving Result ============")
    # args.train_config.clent = args.client
    # args.train_config.server = args.server
    results["config"] = args.train_config  # 只在最后保存一下训练的配置和超参
    results["mean"] = np.mean(results["best_acc"]).item()
    results["stdvar"] = np.var(results["best_acc"]).item()

    std_deviation = math.sqrt(results["stdvar"])
    results["true_stdvar"] = std_deviation
    results["uploadMean"] = readable_size(np.mean(results["upload"]).item())
    results["downloadMean"] = readable_size(np.mean(results["download"]).item())
    results["memoryMean"] = readable_size(np.mean(results["memory"]).item())

    results["allTimeMean"] = np.mean(results["allTime"]).item()
    results["iterationTimeMean"] = np.mean(results["iterationTime"]).item()
    save_result(dataset=args.dataset,
                distribution=args.distribution,
                algorithm=args.algorithm,
                num_clients=args.client["num_clients"],
                result=results,
                prefix=args.prefix,
                debug = args.debug)
    print("============     DONE!     ============")
    afterdraw = False
    if (afterdraw):
        server.drawFeature(bestEpoch=server.bestEpoch)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
    parser.add_argument('--algorithm', type=str, default='FedPSYSSALong', help='name of algorithm')
    # parser.add_argument('--algorithm', type=str, default='FedCPCL', help='name of algorithm')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--prefix', type=str, default='', help='prefix of result filename')
    parser.add_argument('--modelName', type=str, default='LeNet5', help='training model')
    parser.add_argument('--benignAlone',action='store_true', help='if every client has benignAlone')
    parser.add_argument('-m', "--model_family", type=str, default="HtFE2")
    parser.add_argument('--flType', type=str, default="HtFL")
    parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cuda")

    # set_random_seed(42)  # 设置随机种子

    config_file_name = f'./configs/{args.algorithm}.yaml'
    with open(config_file_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("=====================      Config      =====================")
    print(f"Algorithm: {args.algorithm}, Dataset: {args.dataset}, Model: {args.modelName}, Device:{args.device}")
    if args.debug:
        print("debug")
    if args.benignAlone:
        print("benign alone")
    for key, value in config.items():
        print(key + ": " + str(value))
    print("===================== Experiment Begin =====================")

    if args.debug:
        args.num_repeat_times = 1
    else:
        args.num_repeat_times = config["num_repeat_times"]
    args.distribution = config["distribution"]
    args.alpha = config["alpha"]

    args.server = config["server"]
    args.client = config["client"]

    run(args)
