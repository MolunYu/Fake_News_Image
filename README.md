# Fake_News_Image
A fake news image contest

## Directory
```text
Fake_News_Image
├── data
│   ├── model
│   ├── test
│   └── train
├── src
│   ├── bar.py
│   ├── FakeNewsDataset.py
│   ├── pre_process.py
│   ├── __pycache__
│   ├── resume_train.py
│   ├── test.py
│   └── train.py
└── README.md

```

## Baseline
### Resnet50
```text
Test Accuracy of the model on the test images: 90.94 %
Test Precision of the model on the test images: 91.11 %
Test Recall of the model on the test images: 85.69 %
Test F1-score of the model on the test images: 88.32 %
```
### Resnet50 with fft(128 dim)
```text
Test Accuracy of the model on the test images: 91.70 %
Test Precision of the model on the test images: 90.97 %
Test Recall of the model on the test images: 87.97 %
Test F1-score of the model on the test images: 89.44 %
```
### Resnet50 with fft(256 dim by resnet18)
```text
Test Accuracy of the model on the test images: 92.46 %
Test Precision of the model on the test images: 91.58 %
Test Recall of the model on the test images: 89.36 %
Test F1-score of the model on the test images: 90.46 %
```
### Resnet50 with fft(256 dim by resnet18) after adaption
```text
Test Accuracy of the model on the test images: 92.78 %
Test Precision of the model on the test images: 90.92 %
Test Recall of the model on the test images: 91.05 %
Test F1-score of the model on the test images: 90.98 %
```
### Resnet18 * 3 with fft and ela
```text
Test Accuracy of the model on the test images: 93.40 %
Test Precision of the model on the test images: 93.37 %
Test Recall of the model on the test images: 89.88 %
Test F1-score of the model on the test images: 91.59 %

```
### Resnet18 * 3 with fft and ela after adaption
```text
Test Accuracy of the model on the test images: 93.55 %
Test Precision of the model on the test images: 93.93 %
Test Recall of the model on the test images: 89.66 %
Test F1-score of the model on the test images: 91.74 %
```

### Lightgbm fusion fft, ela, cnn
```text
Test Accuracy of the model on the test images: 94.46 %
Test Precision of the model on the test images: 96.09 %
Test Recall of the model on the test images: 94.75 %
Test F1-score of the model on the test images: 95.41 %
```