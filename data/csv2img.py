import os
from argparse import ArgumentParser
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from deep_utils import remove_create, repeat_dimension
import sys
from os.path import split

sys.path.append(split(split(__file__)[0])[0])
from settings import EMOTION_ID2NAME

parser = ArgumentParser()
parser.add_argument("--file_path", help="path to the input file, default ../datasets/fer2013.csv",
                    default="../datasets/fer2013.csv")
parser.add_argument("--train_path", default="../datasets/train")
parser.add_argument("--val_path", default="../datasets/val")
parser.add_argument("--test_path", default="../datasets/test")


def prepare_data(data):
    """ Prepare data for training
        input: data frame with labels and pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def main():
    args = parser.parse_args()
    df = pd.read_csv(args.file_path)
    train_images, train_labels = prepare_data(df[df['Usage'] == 'Training'])
    val_images, val_labels = prepare_data(df[df['Usage'] == 'PublicTest'])
    test_images, test_labels = prepare_data(df[df['Usage'] == 'PrivateTest'])
    save_array_images(args.train_path, EMOTION_ID2NAME, train_images, train_labels)
    save_array_images(args.val_path, EMOTION_ID2NAME, val_images, val_labels)
    save_array_images(args.test_path, EMOTION_ID2NAME, test_images, test_labels)


def save_array_images(root_dir, emotion_mapping, train_images, train_labels):
    remove_create(root_dir)
    for e, (img, lbl) in tqdm(enumerate(zip(train_images, train_labels)), total=len(train_images)):
        name = emotion_mapping[lbl]
        img_dir = os.path.join(root_dir, name)
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"{e:05}.jpg")
        img = repeat_dimension(img)
        cv2.imwrite(img_path, img)


if __name__ == '__main__':
    main()
