from PIL import Image, ImageFile, ImageChops
import numpy as np
from multiprocessing import Pool
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_external_feature(img_path):
    size = os.path.getsize(img_path)
    img = Image.open(img_path)
    l, h = img.size
    return h, l, size, h * l


def log_fft(img_path):
    img = Image.open(img_path).convert("L")
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    f = np.log(np.abs(f) + 1)
    upbound = np.max(f)
    if upbound != 0:
        f = f / upbound * 255
    return Image.fromarray(f).convert("L")


def ela(img_path):
    original = Image.open(img_path).convert("RGB")
    original.save("tmp/tmp_{}.jpg".format(img_path.split("/")[-1].split(".")[0]), quality=90)
    temporary = Image.open("tmp/tmp_{}.jpg".format(img_path.split("/")[-1].split(".")[0]))

    diff = ImageChops.difference(original, temporary)
    d = diff.load()
    width, height = diff.size
    for x in range(width):
        for y in range(height):
            d[x, y] = tuple(k * 10 for k in d[x, y])

    return diff


def trans_fft(path):
    f = log_fft(path)
    pos_neg, path_name, img_name = path.split("/")[-3:]
    if pos_neg in ['train', 'test']:
        f.save("../data/fft/{}/{}/{}".format(pos_neg, path_name, img_name))
    else:
        f.save("../data/fft/{}/{}".format(path_name, img_name))


def trans_ela(path):
    f = ela(path)
    pos_neg, path_name, img_name = path.split("/")[-3:]
    if pos_neg in ['train', 'test']:
        f.save("../data/ela/{}/{}/{}.jpg".format(pos_neg, path_name, img_name.split(".")[0]))
    else:
        f.save("../data/ela/{}/{}.jpg".format(path_name, img_name.split(".")[0]))


if __name__ == '__main__':
    path_list = []

    for root, _, files in os.walk("../data/train/truth_pic"):
        for file in files:
            path_list.append(os.path.join(root, file))

    for root, _, files in os.walk("../data/train/rumor_pic"):
        for file in files:
            path_list.append(os.path.join(root, file))

    for root, _, files in os.walk("../data/test"):
        for file in files:
            if not file.endswith("json"):
                path_list.append(os.path.join(root, file))

    pool = Pool()
    pool.map(trans_ela, path_list)
    pool.close()
    pool.join()
