import torch
import torch.nn as nn
from FakeNewsDataset import FakeNewsDataset
from FakeNewModel import FakeNewsModel
import torch.utils.data as data
import torchvision
import torchvision.transforms as transform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
num_classes = 2
batch_size = 64
learning_rate = 0.001
transform = transform.Compose([transform.Resize((224, 224)), transform.ToTensor()])

train_dataset = FakeNewsDataset(train=True, transform=transform)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

model = FakeNewsModel()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.5)

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels, fourier) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        fourier = fourier.to(device)

        outputs = model(images, fourier)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: [{:.4f}], lr: {}".format(epoch + 1, num_epochs, i + 1, total_step,
                                                                               loss.item(), scheduler.get_lr()))
    scheduler.step()

    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), "../data/model/resnet50_fft_resnet18_epoch{}_lr{}.pth".format(epoch + 1, learning_rate))
