import torch
import torch.nn as nn
import os
from PIL import Image
from FakeNewsDataset import FakeNewsDataset
from FakeNewsModel import FakeNewsModel
import torch.utils.data as data
import torchvision.transforms as transform
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_external_feature(img_path):
    size = os.path.getsize(img_path)
    img = Image.open(img_path)
    l, h = img.size
    return h, l, size, h * l


batch_size = 128
transform = transform.Compose([transform.Resize((224, 224)), transform.ToTensor()])

train_dataset = FakeNewsDataset(train=True, transform=transform)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

test_dataset = FakeNewsDataset(train=False, transform=transform)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = FakeNewsModel()
model.load_state_dict(torch.load("../data/model/resnet18x3_fft_ela_adaption_epoch30.pth"))
model = model.to(device)

# Inference
model.eval()
with torch.no_grad():
    #  Train
    prob_list = []
    label_list = []
    img_list = []

    for images, labels, fourier, ela, img_path in tqdm(train_loader):
        images = images.to(device)
        fourier = fourier.to(device)
        ela = ela.to(device)

        outputs = nn.Softmax(dim=1)(model(images, fourier, ela))
        prob_list.extend(outputs[:, 0].tolist())
        label_list.extend(labels.tolist())

        img_list.extend(img_path)

    external_feature = [get_external_feature(img_path) for img_path in img_list]
    height, length, size, area = zip(*external_feature)

    fusion_feature = pd.DataFrame(
        {"prob": prob_list, "height": height, "length": length, "size": size, "area": area, "label": label_list})
    fusion_feature.to_csv("../data/lgb/train_feature.csv", index=False)

    # Test
    prob_list = []
    label_list = []
    img_list = []

    for images, labels, fourier, ela, img_path in tqdm(test_loader):
        images = images.to(device)
        fourier = fourier.to(device)
        ela = ela.to(device)

        outputs = nn.Softmax(dim=1)(model(images, fourier, ela))
        prob_list.extend(outputs[:, 0].tolist())
        label_list.extend(labels.tolist())

        img_list.extend(img_path)

    external_feature = [get_external_feature(img_path) for img_path in img_list]
    height, length, size, area = zip(*external_feature)

    fusion_feature = pd.DataFrame(
        {"prob": prob_list, "height": height, "length": length, "size": size, "area": area, "label": label_list})
    fusion_feature.to_csv("../data/lgb/test_feature.csv", index=False)
