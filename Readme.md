# FER (Facial Emotion Recognition)

In this repository we present our experiments on fer-2013 dataset. Different models like:
1. VGG

and different augmentation methods like:
2. CutMix 
 
are used in this experiment and more methods are yet to be tried. In case of any errors, please kindly open an issue or 
create a pull request.

## Installation requirements
install the requirements using the following command:
```commandline
pip install -r requirments.txt
```

## Prepare Dataset
Download the official [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset and place it in the dataset folder with the following structure `datasets/fer2013.csv`

The models are compatible with images therefore run the following module to convert `fer2013.csv` dataset to image files.
```commandline
python data/csv2img.py --file_path datasets/fer2013.csv --train_path datasets/train --val_path datasets/val --test_path datasets/test
```


## Training
Each training procedure contains a `yaml` that has the same name of the model to which it belongs, like `vgg.yml` that belongs to the `vgg` model. 
These config files are required for the training and can be easily modified. Furthermore, to make each training procedure pursuable the yaml file is regenerated in the output directory of 
each training.

### VGG
`python train.py --config_path configs/vgg.yml`


## Results

Model | Val ACC | Test Acc 
--- |-------------|-------------|
VGG | 70.8833  | 72.0814


# References
1. https://github.com/usef-kh/fer
