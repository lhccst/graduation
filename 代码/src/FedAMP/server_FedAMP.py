import torch

from src.FedAMP.client_FedAMP import ClientFedAMP
from src.server_base import ServerBase
from mem_utils import MemReporter

class ServerFedAMP(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.alpha = args.server["alpha"]
        self.sigma = args.server["sigma"]

        self.initialize_clients(ClientFedAMP)

    def grad_of_attention(self, wi, wj):
        wi = torch.cat([torch.flatten(param) for param in wi.values()])
        wj = torch.cat([torch.flatten(param) for param in wj.values()])
        delta = (wi - wj).requires_grad_()
        attention = 1 - torch.exp(-torch.norm(delta)**2 / self.sigma)
        attention.backward()
        grad = torch.dot(delta.grad, delta.grad)
        return grad

    def receive_aggregated_dispatch(self):
        client_models = []
        for client in self.join_clients:
            client_models.append(client.model.state_dict())
            self.reporter.track_upload(client.model.state_dict())

        attentions = torch.zeros(len(self.join_clients), len(self.join_clients))

        for i in range(len(self.join_clients)):
            for j in range(i+1, len(self.join_clients)):
                attentions[i][j] = self.grad_of_attention(client_models[i], client_models[j])
                attentions[j][i] = attentions[i][j]

            attentions[i][i] = 1 - torch.sum(attentions[i])



        for k in range(len(self.join_clients)):
            attention = attentions[k].to(self.device)

            aggregated = {}
            for key in self.args.model.state_dict().keys():
                param = [client_model[key] for client_model in client_models]
                param = torch.stack(param, dim=0)
                shape = [-1] + [1] * (len(param.shape) - 1)
                aggregated[key] = torch.sum(param * attention.reshape(shape), dim=0)

            self.join_clients[k].global_model = aggregated
            self.reporter.track_download(aggregated)


    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()

            self.receive_aggregated_dispatch()

            print("================== evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory
