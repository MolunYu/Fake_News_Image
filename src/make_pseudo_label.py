import torch
import torch.nn as nn
from FakeNewsDataset import FakeNewsDataset
from FakeNewsModel import FakeNewsModel
import torch.utils.data as data
import torchvision.transforms as transform
from tqdm import tqdm
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 64
transform = transform.Compose([transform.Resize((224, 224)), transform.ToTensor()])

test_dataset = FakeNewsDataset(train=False, transform=transform)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = FakeNewsModel()
model.load_state_dict(torch.load("../data/model/resnet18x3_fft_ela_epoch60.pth"))
model = model.to(device)

img_path_list = []
label_list = []
score_list = []

# Test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    TP = FP = FN = 0

    for images, _, fourier, ela, img_paths in tqdm(test_loader):
        images = images.to(device)
        fourier = fourier.to(device)
        ela = ela.to(device)

        outputs = nn.Softmax(dim=1)(model(images, fourier, ela))
        scores, predicted = torch.max(outputs.detach(), 1)
        img_path_list.extend(img_paths)
        label_list.append(predicted)
        score_list.append(scores)

labels = torch.cat(label_list, 0).tolist()
scores = torch.cat(score_list, 0).tolist()

result = list(zip(img_path_list, labels, scores))
result = sorted(result, key=lambda x: -x[2])
threshold = round(len(result) * 0.9)
img2label = dict()

for img_path, label, _ in result[:threshold]:
    img2label[img_path] = label

with open("../data/pseudo/img2label.json", mode="w") as dst:
    json.dump(img2label, dst)
