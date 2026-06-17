import torch
import numpy as np

from src.FPL.client_FPL import ClientFPL
from src.server_base import ServerBase
from src.FPL.FINCH import FINCH
from mem_utils import MemReporter


class ServerFPL(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.local_protos = {}
        self.aggregated = self.args.model.base.state_dict()

        self.initialize_clients(ClientFPL)

    def dispatch(self):
        for client in self.join_clients:
            client.model.base.load_state_dict(self.aggregated)
            client.global_protos = self.global_protos
            self.reporter.track_download(self.aggregated)
            self.reporter.track_download(self.global_protos)

    def receive_and_aggregated(self):
        client_models = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_models.append(client.model.base.state_dict())
            client_sample_sizes.append(client.sample_size)
            self.reporter.track_upload(client.model.base.state_dict())
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        self.reporter.track_upload(size_tensor)
        client_weight = size_tensor / torch.sum(size_tensor)

        for key in self.aggregated.keys():
            param = [client_model[key] for client_model in client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.aggregated[key] = torch.sum(param * client_weight.reshape(shape), dim=0)

        self.global_protos = self.proto_aggregation([client.local_protos for client in self.join_clients])

    def proto_aggregation(self, local_protos_list):
        self.reporter.track_upload(local_protos_list)
        agg_protos_label = dict()
        for idx in range(len(local_protos_list)):
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_and_aggregated()
            self.dispatch()

            print("================== evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory




