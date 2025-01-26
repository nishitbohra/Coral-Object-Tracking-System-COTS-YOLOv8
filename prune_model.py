import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r'E:\COTS\aquarium_pretrain\src\runs\cots\weights\best.pt'

model = YOLO(model_path)  

model.model.to(device)  

pruning_amount = 0.2  
for name, module in model.model.named_modules():  
    if isinstance(module, torch.nn.Conv2d):  
        prune.l1_unstructured(module, name="weight", amount=pruning_amount)

pruned_model_path = 'E:/COTS/aquarium_pretrain/src/runs/cots/weights/pruned_best.pt'
model.save(pruned_model_path)  

print(f"Pruned model saved at: {pruned_model_path}")
