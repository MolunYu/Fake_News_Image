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
            self.ela_path = "../data/ela/train"
        else:
            json_path = "../data/test/img2label.json"
            self.fft_path = "../data/fft/test"
            self.ela_path = "../data/ela/test"

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
            ela = Image.open(os.path.join(self.ela_path, path_name, "{}.jpg".format(img_name.split('.')[0])))
        else:
            fourier = Image.open(os.path.join(self.fft_path, img_name))
            ela = Image.open(os.path.join(self.ela_path, "{}.jpg".format(img_name.split('.')[0])))

        if self.transform:
            img = self.transform(img)
            fourier = self.transform(fourier)
            ela = self.transform(ela)

        return img, int(label), fourier, ela

    def __len__(self) -> int:
        return len(self.img_label)


class PseudoDataset(data.Dataset):
    def __init__(self, transform):
        json_path = "../data/pseudo/img2label.json"
        self.fft_path = "../data/fft/test"
        self.ela_path = "../data/ela/test"

        with open(json_path, mode="r") as src:
            self.img_label = tuple(json.load(src).items())

        self.transform = transform

    def __getitem__(self, index: int):
        img_path, label = self.img_label[index]
        img = Image.open(img_path).convert("RGB")
        path_name, img_name = img_path.split("/")[-2:]
        fourier = Image.open(os.path.join(self.fft_path, img_name))
        ela = Image.open(os.path.join(self.ela_path, "{}.jpg".format(img_name.split('.')[0])))

        if self.transform:
            img = self.transform(img)
            fourier = self.transform(fourier)
            ela = self.transform(ela)

        return img, int(label), fourier, ela

    def __len__(self) -> int:
        return len(self.img_label)
