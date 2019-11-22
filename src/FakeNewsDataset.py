import torch
import json
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FakeNewsDataset(data.Dataset):
    def __init__(self, train=True, transform=None) -> None:
        if train:
            json_path = "../data/train/img2label.json"
        else:
            json_path = "../data/test/img2label.json"

        with open(json_path, mode="r") as src:
            self.img_label = tuple(json.load(src).items())

        self.transform = transform

    def __getitem__(self, index: int):
        img_path, label = self.img_label[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, int(label)

    def __len__(self) -> int:
        return len(self.img_label)
