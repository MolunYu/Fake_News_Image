from PIL import Image
import numpy as np
from multiprocessing import Pool
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


def log_fft(img_path):
    img = Image.open(img_path).convert("L")
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    f = np.log(np.abs(f) + 1)
    upbound = np.max(f)
    if upbound != 0:
        f = f / upbound * 255
    return Image.fromarray(f).convert("L")


def trans_fft(path):
    f = log_fft(path)
    pos_neg, path_name, img_name = path.split("/")[-3:]
    if pos_neg in ['train', 'test']:
        f.save("../data/fft/{}/{}/{}".format(pos_neg, path_name, img_name))
    else:
        f.save("../data/fft/{}/{}".format(path_name, img_name))


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
    pool.map(trans_fft, path_list)
    pool.close()
    pool.join()
