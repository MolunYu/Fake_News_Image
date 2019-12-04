from tsnecuda import TSNE
from matplotlib import pyplot as plt
import torch
from FakeNewsDataset import FakeNewsDataset
from FakeNewModel import FakeNewsModel
import torch.utils.data as data
import torchvision.transforms as transform
from tqdm import tqdm
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 64
transform = transform.Compose([transform.Resize((224, 224)), transform.ToTensor()])

test_dataset = FakeNewsDataset(train=False, transform=transform)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = FakeNewsModel()
model.load_state_dict(torch.load("../data/model/resnet50_fft_resnet18_epoch80_lr0.001.pth"))
model = model.to(device)
feature_list = []
label_list = []

# Test
model.eval()
with torch.no_grad():

    for images, labels, fourier in tqdm(test_loader):
        images = images.to(device)
        fourier = fourier.to(device)
        labels = labels.to(device)

        _, feature = model(images, fourier)
        feature_list.append(feature)
        label_list.append(labels)

features = torch.Tensor.cpu(torch.cat(feature_list, 0))
X = TSNE(n_components=2).fit_transform(features)
labels = torch.Tensor.cpu(torch.cat(label_list, 0))

plt.figure()
for i in tqdm(range(len(X))):
    if labels[i].item() == 0:
        plt.plot(X[i, 0], X[i, 1], 'r.')
    else:
        plt.plot(X[i, 0], X[i, 1], 'g.')
plt.savefig("../data/visualize/{}.jpg".format(time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())))




