# import numpy as np
# import torch
# from sklearn.cluster import KMeans
#
# from src.FedCPCL.client_FedCPCL import ClientFedCPCL
# from src.server_base import ServerBase
#
#
# class ServerFedCPCL(ServerBase):
#     def __init__(self, args):
#         super().__init__(args)
#
#         self.sample_weights = None
#         self.initialize_clients(ClientFedCPCL)
#
#         self.global_protos = None
#         self.received_protos = []
#
#     def dispatch(self):
#         for client in self.join_clients:
#             client.global_protos = self.global_protos
#
#     def receive(self):
#         self.received_protos = []
#         sample_sizes = []
#         for client in self.join_clients:
#             protos = []
#             sizes = []
#             for key in client.local_protos.keys():
#                 protos.append(client.local_protos[key])
#                 sizes.append(client.statistic[str(key)])
#
#             self.received_protos.extend(protos)
#             sample_sizes.extend(sizes)
#
#         self.sample_weights = np.array(sample_sizes) / np.sum(sample_sizes)
#         self.sample_weights *= len(self.received_protos)
#
#     def aggregate(self):
#         if len(self.received_protos) < 100:
#             return
#
#         proto_tensor = torch.stack(self.received_protos).cpu()
#         proto_np_array = proto_tensor.numpy()
#
#         if self.args.server["clustering"] == "kmeans":
#             clustering = KMeans(n_clusters=self.num_classes, algorithm="elkan")
#             clustering.fit(proto_np_array, sample_weight=self.sample_weights)
#
#             centroids = clustering.cluster_centers_  # 获取聚类中心
#
#         else:
#             raise NotImplementedError
#
#         self.global_protos = [torch.from_numpy(centroid).to(self.device) for centroid in centroids]
#
#     def train(self):
#         acc_record = {}
#         for i in range(self.global_rounds):
#             self.join_clients = self.select_join_clients()
#             print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")
#
#             for client in self.join_clients:
#                 client.train()
#
#             self.receive()
#             self.aggregate()
#             self.dispatch()
#
#             print("================= evaluate =================")
#             acc = self.evaluate()
#             acc_record[i] = acc
#             print(f"================= global round: {i}, accuracy: {acc} =================")
#
#         return acc_record



import numpy as np
import torch
from sklearn.cluster import KMeans
from mem_utils import MemReporter
from src.FedCPCL.client_FedCPCL import ClientFedCPCL
from src.server_base import ServerBase
# from src.FedCPCL.finch import FINCH
from sklearn.cluster import DBSCAN
class ServerFedCPCL(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.sample_weights = None
        self.initialize_clients(ClientFedCPCL)

        self.global_protos = None
        self.received_protos = []

    def dispatch(self):
        for client in self.join_clients:
            self.reporter.track_download(self.global_protos)
            client.global_protos = self.global_protos

    def receive(self):
        self.received_protos = []
        self.received_protos_class = []
        sample_sizes = []
        proto_weights_sizes =[]

        for client in self.join_clients:
            protos = []
            protosClass =[]
            sizes = []
            weights = []
            for key in client.local_protos.keys():
                protos.append(client.local_protos[key])
                self.reporter.track_upload(client.local_protos[key])
                protosClass.append(key)
                sizes.append(client.statistic[str(key)])
                self.reporter.track_upload(client.statistic[str(key)])
                weights.append(client.proto_weight[key])
                self.reporter.track_upload(client.proto_weight[key])

            self.received_protos.extend(protos)
            self.received_protos_class.extend(protosClass)
            sample_sizes.extend(sizes)
            proto_weights_sizes.extend(weights)




        self.sample_weights = np.array(sample_sizes) / np.sum(sample_sizes)
        self.proto_weights = np.array(proto_weights_sizes) / np.sum(proto_weights_sizes)
        # print('sample_weights:',self.sample_weights)
        # self.proto_weights = np.array(proto_weights_sizes)
        #
        self.proto_weights += self.sample_weights
        self.proto_weights *= len(self.received_protos)
        self.sample_weights *= len(self.received_protos)
        # self.proto_weights *= self.sample_weights
        # print("proto_weight:", self.proto_weights)

    def aggregate(self):
        # if len(self.received_protos) < 100:
        #     return

        proto_tensor = torch.stack(self.received_protos).cpu()
        proto_np_array = proto_tensor.numpy()
        # if proto_np_array.shape[0]>self.num_classes*2:
        #     proto_weights_tensor = torch.from_numpy(self.proto_weights)
        #     threshold = torch.quantile(proto_weights_tensor, 0.5)
        #     mask = proto_weights_tensor >= threshold
        #     proto_np_array = proto_np_array[mask]
        #     self.proto_weights = self.proto_weights[mask]
        # # print("一般原型置信度：",self.proto_weights)
        #     print('筛选后的proto数量：',proto_np_array.shape[0])

        if self.args.server["clustering"] == "kmeans":
            clustering = KMeans(n_clusters=self.num_classes, algorithm="elkan")
            clustering.fit(proto_np_array, sample_weight=self.proto_weights)

            centroids = clustering.cluster_centers_  # 获取聚类中心

            self.global_protos = [torch.from_numpy(centroid).to(self.device) for centroid in centroids]
        elif self.args.server["clustering"] == "finch":
            print("finch")
            c, num_clust, req_c = FINCH(proto_np_array, initial_rank=None, req_clust=None, distance='cosine',
                                        ensure_early_exit=False, verbose=True)
            m, n = c.shape
            class_cluster_list = []
            for index in range(m):
                class_cluster_list.append(c[index, -1])

            class_cluster_array = np.array(class_cluster_list)
            unique_cluster = np.unique(class_cluster_array).tolist()
            agg_selected_proto = []

            for _, cluster_index in enumerate(unique_cluster):
                selected_array = np.where(class_cluster_array == cluster_index)
                selected_proto_list = proto_np_array[selected_array]
                proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                agg_selected_proto.append(proto)

            # centroids = np.array(agg_selected_proto)
            self.global_protos = [torch.from_numpy(np.squeeze(centroid)).to(self.device) for centroid in agg_selected_proto]

        elif self.args.server["clustering"] == "dbscan":
            print("DBSCAN")
            dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')  # 根据需求调整参数
            dbscan.fit(proto_np_array)

            labels = dbscan.labels_
            unique_labels = np.unique(labels)
            agg_selected_proto = []

            for label in unique_labels:
                if label == -1:
                    continue  # 跳过噪声点

                selected_array = np.where(labels == label)
                selected_proto_list = proto_np_array[selected_array]
                proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                agg_selected_proto.append(proto)

            self.global_protos = [torch.from_numpy(np.squeeze(centroid)).to(self.device) for centroid in agg_selected_proto]

        else:
            raise NotImplementedError


    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive()
            self.aggregate()
            self.dispatch()

            print("================= evaluate =================")
            acc = self.evaluate(global_epoch=i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")
            # self.reporter.print_communication_stats()

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory
