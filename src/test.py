import torch
from FakeNewsDataset import FakeNewsDataset
from FakeNewModel import FakeNewsModel
import torch.utils.data as data
import torchvision.transforms as transform
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 64
transform = transform.Compose([transform.Resize((224, 224)), transform.ToTensor()])

test_dataset = FakeNewsDataset(train=False, transform=transform)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = FakeNewsModel()
model.load_state_dict(torch.load("../data/model/resnet50_fft_epoch100_lr0.001.pth"))
model = model.to(device)

# Test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    TP = FP = FN = 0

    for images, labels, fourier in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        fourier = fourier.to(device)

        outputs = model(images, fourier)
        _, predicted = torch.max(outputs.detach(), 1)
        correct += torch.sum(predicted == labels).item()
        total += images.size(0)

        result = ["{}{}".format(i, j) for i, j in zip(predicted, labels)]
        TP += result.count("00")
        FP += result.count("01")
        FN += result.count("10")

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print('Test Accuracy of the model on the test images: {:.2f} %'.format(100 * correct / total))
    print('Test Precision of the model on the test images: {:.2f} %'.format(100 * precision))
    print('Test Recall of the model on the test images: {:.2f} %'.format(100 * recall))
    print('Test F1-score of the model on the test images: {:.2f} %'.format(
        100 * 2 * precision * recall / (precision + recall)))
