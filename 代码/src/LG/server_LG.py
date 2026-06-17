import torch
from src.LG.client_LG import ClientLG
from src.server_base import ServerBase


class ServerLG(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.global_head = self.args.head.state_dict()
        self.initialize_clients(ClientLG)

    def dispatch(self):
        """仅分发全局头部参数"""
        for client in self.join_clients:
            self.reporter.track_download(self.global_head)
            client.model.head.load_state_dict(self.global_head)

    def receive_and_aggregated(self):
        client_heads = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_heads.append(client.model.head.state_dict())
            client_sample_sizes.append(client.sample_size)
            self.reporter.track_upload(client.model.head.state_dict())
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        self.reporter.track_upload(size_tensor)
        client_weight = size_tensor / torch.sum(size_tensor)

        for key in self.global_head.keys():
            param = [client_head[key] for client_head in client_heads]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.global_head[key] = torch.sum(param * client_weight.reshape(shape), dim=0)


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
            acc = self.evaluate(i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory
