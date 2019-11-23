import torch
import torch.nn as nn
from FakeNewsDataset import FakeNewsDataset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 30
num_classes = 2
batch_size = 64
learning_rate = 0.0001
transform = transform.Compose([transform.Resize((224, 224)), transform.ToTensor()])

train_dataset = FakeNewsDataset(train=True, transform=transform)
test_dataset = FakeNewsDataset(train=False, transform=transform)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("../data/model/params_epoch100.ckpt"))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch, num_epochs, i + 1, total_step, loss.item()))

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), "../data/model/params_epoch{}_lr{}.ckpt".format(epoch + 1, learning_rate))

# Test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.detach(), 1)
        correct += torch.sum(predicted == labels).item()
        total += images.size(0)

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

