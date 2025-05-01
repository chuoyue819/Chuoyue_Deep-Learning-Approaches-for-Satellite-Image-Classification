import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from timm import create_model
from PIL import Image
from torch.utils.data import Dataset
import json

model_root = '/Users/wangchuoyue/Desktop/final project/model_10class_rexnet150'
new_image_root = '/Users/wangchuoyue/Desktop/final project/new_image_city'

print('using model from root:', model_root)
model_info_path = os.path.join(model_root, 'model_info.txt')
label_index_to_name_path = os.path.join(model_root, 'label_index_to_name.txt')
model_file_path = os.path.join(model_root, 'model')

with open(model_info_path) as f:
    print(f.read())
print('label information:')
with open(label_index_to_name_path) as f:
    print(f.read())
with open(label_index_to_name_path) as f:
    label_index_to_name = json.load(f)
label_index_to_name = {int(k): v for k, v in label_index_to_name.items()}
print()
model = torch.load(model_file_path, weights_only=False)

class NoLabelImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.endswith(('jpg', 'jpeg', 'png'))]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = NoLabelImageDataset(root_dir=new_image_root, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False)

model.eval()
with torch.no_grad():
    for images, paths in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for p_index, path in zip(predicted, paths):
            print(path, label_index_to_name[p_index.item()])
