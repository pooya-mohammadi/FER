## FER (Facial Emotion Recognition)

In this repository we present our experiments on fer-2013 dataset


# Download Dataset
Download the official [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset, and place it in the dataset folder with the following folder structure `datasets/fer2013.csv`

# Installation
`pip install -r requirments.txt`

# Train
<a href="https://colab.research.google.com/github/pooya-mohammadi/FER/blob/master/notebooks/train.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
## VGG
`python train.py --model-name vgg --crop-size 40 --batch-size 8 --n-workers 4`


## Results

Model | Val ACC | Test Acc 
--- |-------------|-------------|
VGG | 70.8833  | 72.0814


# References
1. https://github.com/usef-kh/fer
