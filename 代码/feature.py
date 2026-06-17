import torch

best_global_epoch =31
best_model_path= "/root/autodl-tmp/FedPSY226/results/kronoDroid/models/"+str(best_global_epoch)
model_path = best_model_path+"/"+str(clientSave.client_id)+".pth"

# 加载最佳模型
best_model_path = "best_model.pth"
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # 设置为评估模式