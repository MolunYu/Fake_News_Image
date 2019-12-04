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