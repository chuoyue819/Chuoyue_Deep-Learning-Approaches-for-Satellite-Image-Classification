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
import json

ori_data_root = '/Users/wangchuoyue/Desktop/final project/Aerial_Landscapes'
organized_data_root = '/Users/wangchuoyue/Desktop/final project/Aerial_Landscapes11.organized'
model_save_root = '/Users/wangchuoyue/Desktop/final project/model'
train_data_portion = 0.8
validation_data_portion = 0.1

model_name = 'pretrained rexnet150'
supported_models = [
    'logistic regression',
    'squeezenet1_0',
    'mobilenet v2',
    'resnet18',
    'vit',
    'pretrained vit',
    'rexnet150',
    'pretrained rexnet150'
]
number_of_epoch = 5
learning_rate_base = 1e-3
batch_size = 64

print('------data information------')
print('ori data root:', ori_data_root)
print('organized data root:', organized_data_root)
number_of_class = len(os.listdir(ori_data_root))
print('train data portion:', train_data_portion)
print('number of class:', number_of_class)

if not os.path.exists(model_save_root):
    os.mkdir(model_save_root)
if os.path.exists(organized_data_root):
    print('organized data root exists. data has already been organized.')
    print('if things fail, maybe try deleting organized data root and re-run.')
else:
    print('starting to organize data...')
    os.mkdir(organized_data_root)
    train_root = os.path.join(organized_data_root, 'train')
    validation_root = os.path.join(organized_data_root, 'validation')
    test_root = os.path.join(organized_data_root, 'test')
    os.mkdir(train_root)
    os.mkdir(validation_root)
    os.mkdir(test_root)

    for sub_dir in os.listdir(ori_data_root):
        if sub_dir.startswith('.'):
            continue
        full_path_of_sub_dir = os.path.join(ori_data_root, sub_dir)
        full_path_of_train_sub_dir = os.path.join(train_root, sub_dir)
        full_path_of_validation_sub_dir = os.path.join(validation_root, sub_dir)
        full_path_of_test_sub_dir = os.path.join(test_root, sub_dir)
        os.mkdir(full_path_of_train_sub_dir)
        os.mkdir(full_path_of_validation_sub_dir)
        os.mkdir(full_path_of_test_sub_dir)

        all_files = os.listdir(full_path_of_sub_dir)
        sorted_files = sorted(all_files)
        train_num_in_each_category = int(train_data_portion * len(all_files))
        validation_num_in_each_category = train_num_in_each_category + int(validation_data_portion * len(all_files))
        for i in range(len(all_files)):
            full_file_path = os.path.join(full_path_of_sub_dir, sorted_files[i])
            is_train_data = i < train_num_in_each_category
            is_validation_data = i >= train_num_in_each_category and i < validation_num_in_each_category
            if is_train_data:
                full_organized_file_path = os.path.join(full_path_of_train_sub_dir, sorted_files[i])
            elif is_validation_data:
                full_organized_file_path = os.path.join(full_path_of_validation_sub_dir, sorted_files[i])
            else:
                full_organized_file_path = os.path.join(full_path_of_test_sub_dir, sorted_files[i])
            shutil.copy(full_file_path, full_organized_file_path)
    print('finish organizing data')

print('------model training information------')
print('model name:', model_name)
if model_name not in supported_models:
    print(model_name, 'is not supported.')
    print('supported models:', supported_models)
    print('pick model from supported, and re-run')
    os.sys.exit(1)
print('related parameters')
print('  number of epoch:', number_of_epoch)
print('  learning rate base:', learning_rate_base)
print('  batch size:', batch_size)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, number_of_class):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, number_of_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


if model_name == 'logistic regression':
    input_dim = 3 * 224 * 224
    model = LogisticRegressionModel(input_dim, number_of_class)
elif model_name == 'squeezenet1_0':
    model = models.squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, number_of_class, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = number_of_class
elif model_name == 'mobilenet v2':
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, number_of_class)
elif model_name == 'resnet18':
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, number_of_class)
elif model_name == 'vit':
    model = create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, number_of_class)
elif model_name == 'pretrained vit':
    model = create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, number_of_class)
elif model_name == 'rexnet150':
    model = create_model("rexnet_150", pretrained=False)
    model.head.fc = nn.Linear(model.num_features, number_of_class)
elif model_name == 'pretrained rexnet150':
    model = create_model("rexnet_150", pretrained=True)
    model.head.fc = nn.Linear(model.num_features, number_of_class)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = datasets.ImageFolder(root=organized_data_root + "/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = datasets.ImageFolder(root=organized_data_root + "/validation", transform=transform)
validation_loader = DataLoader(validation_dataset, shuffle=False)
test_dataset = datasets.ImageFolder(root=organized_data_root + "/test", transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate_base)
current_time = datetime.now().strftime("%H:%M:%S")
print(current_time, 'starting to train...')
for epoch in range(number_of_epoch):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    current_time = datetime.now().strftime("%H:%M:%S")
    epoch_info = f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}"
    print(current_time, epoch_info)


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    all_pred = []
    all_label = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_pred.extend(predicted)
            all_label.extend(labels)
    return correct, total


print('evaluating on validation data...')
validation_correct, validation_total = evaluate_model(model, validation_loader)
print('  total instances:', validation_total)
print('  correct instances:', validation_correct)
print(f'validation accuracy: {100 * validation_correct / validation_total:.2f}%')

print('evaluating on test data...')
test_correct, test_total = evaluate_model(model, test_loader)
print('  total instances:', test_total)
print('  correct instances:', test_correct)
print(f'test accuracy: {100 * test_correct / test_total:.2f}%')

model_info_path = os.path.join(model_save_root, 'model_info.txt')
with open(model_info_path, 'w') as f:
    print('ori data root:', ori_data_root, file=f)
    print('organized data root:', organized_data_root, file=f)
    print('train data portion:', train_data_portion, file=f)
    print('number of class:', number_of_class, file=f)
    print('model name:', model_name, file=f)
    print('  number of epoch:', number_of_epoch, file=f)
    print('  learning rate base:', learning_rate_base, file=f)
    print('  batch size:', batch_size, file=f)
    print(f'validation accuracy: {100 * validation_correct / validation_total:.2f}%', file=f)
    print(f'test accuracy: {100 * test_correct / test_total:.2f}%', file=f)

label_index_to_name_path = os.path.join(model_save_root, 'label_index_to_name.txt')
label_index_to_name = {v: k for k, v in train_dataset.class_to_idx.items()}
with open(label_index_to_name_path, 'w') as f:
    json.dump(label_index_to_name, f, indent=4)

model_file_path = os.path.join(model_save_root, 'model')
torch.save(model, model_file_path)

print('model saved at root:', model_save_root)

