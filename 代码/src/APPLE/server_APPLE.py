from src.APPLE.client_APPLE import ClientAPPLE
from src.server_base import ServerBase
from mem_utils import MemReporter

class ServerAPPLE(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_clients(ClientAPPLE)

        initial_param = self.args.model.state_dict()
        self.client_models = [initial_param for _ in range(self.num_clients)]

    def dispatch(self, client_models):
        for client in self.join_clients:
            for client_id in range(self.num_clients):
                for layer in self.args.model.state_dict().keys():
                    client.agg_model.client_models[client_id][layer].detach().copy_(client_models[client_id][layer])
                    self.reporter.track_download(client_models[client_id][layer])

    def receive(self):
        for client in self.join_clients:
            self.client_models[client.client_id] = client.agg_model.client_models[client.client_id]
            self.reporter.track_upload(client.agg_model.client_models[client.client_id])

    def train(self):
        acc_record = {}
        for i in range(self.global_rounds):
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            self.dispatch(self.client_models)

            for client in self.join_clients:
                client.train()

            self.receive()

            print("================= evaluate =================")
            acc = self.evaluate()
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom,downloadCom,usedMemory =self.getReport()
        return acc_record,uploadCom,downloadCom,usedMemory
