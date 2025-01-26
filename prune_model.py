import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO  # YOLOv8 import

# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model directly from the YOLOv8 framework
model_path = r'E:\COTS\aquarium_pretrain\src\runs\cots\weights\best.pt'

# Load the model (with Ultralytics YOLOv8, no need to access 'model' key, just load it directly)
model = YOLO(model_path)  # This returns a full YOLOv8 model, not just state_dict

# Move the model to the correct device
model.model.to(device)  # Accessing model's internal model

# Prune the model (remove specified percentage of the weights in each layer)
pruning_amount = 0.2  # Dynamic pruning amount
for name, module in model.model.named_modules():  # Accessing the layers inside the model
    if isinstance(module, torch.nn.Conv2d):  # Focus on Conv2d layers for pruning
        prune.l1_unstructured(module, name="weight", amount=pruning_amount)

# Save the pruned model
pruned_model_path = 'E:/COTS/aquarium_pretrain/src/runs/cots/weights/pruned_best.pt'
model.save(pruned_model_path)  # Save using YOLOv8's save method

print(f"Pruned model saved at: {pruned_model_path}")
