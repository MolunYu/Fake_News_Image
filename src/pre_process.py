import os
import json
import random
import shutil

test = list()
img2label_train = dict()
img2label_test = dict()

for root, _, files in os.walk("../data/train/truth_pic"):
    files_path = [os.path.join(root, file) for file in files]
    test.extend(random.sample(files_path, len(files) // 10))

    for file_path in files_path:
        if file_path not in test:
            img2label_train[file_path] = 1
        else:
            img2label_test["../data/test/{}".format(str(file_path).split("/")[-1])] = 1

for root, _, files in os.walk("../data/train/rumor_pic"):
    files_path = [os.path.join(root, file) for file in files]
    test.extend(random.sample(files_path, len(files) // 10))

    for file_path in files_path:
        if file_path not in test:
            img2label_train[file_path] = 0
        else:
            img2label_test["../data/test/{}".format(str(file_path).split("/")[-1])] = 0

with open("../data/train/img2label.json", mode="w") as dst:
    json.dump(img2label_train, dst)

with open("../data/test/img2label.json", mode="w") as dst:
    json.dump(img2label_test, dst)

for i in test:
    shutil.move(i, "../data/test/{}".format(str(i).split("/")[-1]))
