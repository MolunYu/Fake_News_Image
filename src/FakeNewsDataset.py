import json
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FakeNewsDataset(data.Dataset):
    def __init__(self, train=True, transform=None) -> None:
        if train:
            json_path = "../data/train/img2label.json"
            self.fft_path = "../data/fft/train"
        else:
            json_path = "../data/test/img2label.json"
            self.fft_path = "../data/fft/test"

        with open(json_path, mode="r") as src:
            self.img_label = tuple(json.load(src).items())

        self.transform = transform
        self.train = train

    def __getitem__(self, index: int):
        img_path, label = self.img_label[index]
        img = Image.open(img_path).convert("RGB")
        path_name, img_name = img_path.split("/")[-2:]
        if self.train:
            fourier = Image.open(os.path.join(self.fft_path, path_name, img_name))
        else:
            fourier = Image.open(os.path.join(self.fft_path, img_name))

        if self.transform:
            img = self.transform(img)
            fourier = self.transform(fourier)

        return img, int(label), fourier

    def __len__(self) -> int:
        return len(self.img_label)
