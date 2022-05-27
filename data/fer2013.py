import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import cv2
from deep_utils import crawl_directory_dataset, log_print, value_error_log, CutMixTorch
from data.augmetations import get_augmentation
from settings import EMOTION_NAME2ID
from utils.config_utils import Config


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, n_classes=7, logger=None, verbose=1):
        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.transform = transform
        log_print(logger, f"Successfully created {self.__class__.__name__}, samples: {len(self)}", verbose=verbose)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_address = self.images[idx]
        img = cv2.imread(img_address)[..., ::-1]  # BGR2RGB
        img = self.transform(image=img)['image']
        label = torch.tensor(self.labels[idx]).type(torch.long)
        label = F.one_hot(label, num_classes=self.n_classes)
        sample = (img, label)

        return sample

    def collate_fn(self, data):
        images = torch.concat([d[0].unsqueeze(0) for d in data])
        labels = torch.concat([d[1].unsqueeze(0) for d in data])
        cutmix_images, cutmix_labels = CutMixTorch.cls_cutmix_batch(a_images=images, a_labels=labels)
        return cutmix_images, cutmix_labels


# class RESCustomDataset(Dataset):
#     mu, st = 0, 255
#
#     def __init__(self, category, data, image_size=224, number_of_test=10,
#                  augment=True, NoF=False, rotation_degree=0, n_channel=1, **kwargs):
#         self.category = category
#         self.data = data
#         self.n_channel = n_channel
#         self.pixels = self.data['pixels'].tolist()
#         self.emotions = pd.get_dummies(self.data['emotion'])
#         self.augment = augment
#         self.test_number = number_of_test
#         self.NoF = NoF
#         self.image_size = (image_size, image_size)
#         self.applytransform = True
#         # self.aug = iaa.Sequential(
#         #     [iaa.Fliplr(p=0.5),
#         #      iaa.Affine(rotate=(-30, 30))]
#         # )
#         self.basetransform = transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.ToTensor()
#             ]
#         )
#         if NoF:
#             self.testtransform = transforms.Compose(
#                 [transforms.ToPILImage(),
#                  transforms.Pad(2),
#                  transforms.TenCrop(kwargs['crop_size']),
#                  transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#                  ]
#             )
#         else:
#             self.testtransform = self.basetransform
#
#         if augment:
#             self.traintransform = transforms.Compose(
#                 [transforms.ToPILImage(),
#                  transforms.RandomRotation(rotation_degree),
#                  # transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
#                  transforms.RandomHorizontalFlip(),
#                  # transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5 if kwargs["gussian_blur"] else 0),
#                  transforms.Pad(2),
#                  transforms.TenCrop(kwargs["crop_size"]),
#                  transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#                  # transforms.Lambda(
#                  #     lambda tensors: torch.stack(
#                  #         [transforms.Normalize(mean=(self.mu,), std=(self.st,))(t) for t in tensors])),
#                  # transforms.Lambda(
#                  #     lambda tensors: torch.stack(
#                  #         [transforms.RandomErasing(p=0 if kwargs["cutmix"] else 0.5)(t) for t in tensors])),
#                  ]
#             )
#         else:
#             self.traintransform = self.basetransform
#
#     def __len__(self):
#         return len(self.pixels)
#
#     def __getitem__(self, idx):
#         pixels = self.pixels[idx]
#         pixels = list(map(int, pixels.split(" ")))
#         image = np.reshape(pixels, (48, 48)).astype(np.uint8)  # the pixels are in 48*48. the static set is correct
#         image = cv2.resize(image, self.image_size)
#         image = np.dstack([image] * self.n_channel)
#         if self.category == 'train':
#             # if self.augment:
#             #     image = self.aug(image=image)
#             image = self.traintransform(image)
#             # target = torch.tensor(self.emotions.iloc[idx].idxmax())
#             # return image, target
#         if self.category == "test":
#             image = self.testtransform(image)
#         # if self.NoF:
#         #     images = [self.aug(image=image) for i in range(self.test_number)]
#         #     images = [image for i in range(self.test_number)]
#         #     images = list(map(self.testtransform, images))
#         #     target = self.emotions.iloc[idx].idxmax()
#         #     return images, target
#         # else:
#         #     image = self.aug(image=image)
#
#         if self.category == "val":
#             image = self.basetransform(image)
#
#         target = torch.tensor(self.emotions.iloc[idx].idxmax())
#         return image, target


def load_data(path='datasets/fer2013/fer2013.csv'):
    fer2013 = pd.read_csv(path)
    EMOTION_NAME2IDping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, EMOTION_NAME2IDping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def get_data_loaders(config: Config, dataloader_name=None, logger=None, verbose=1):
    train_transform, val_transform, test_transform = get_augmentation(config.augmentation.name,
                                                                      img_h=config.dataset.img_h,
                                                                      img_w=config.dataset.img_w,
                                                                      mean=config.augmentation.mean,
                                                                      std=config.augmentation.std)

    train_address, train_labels = crawl_directory_dataset(config.dataset.train_path, label_map_dict=EMOTION_NAME2ID,
                                                          logger=logger, verbose=verbose)
    val_address, val_labels = crawl_directory_dataset(config.dataset.val_path, label_map_dict=EMOTION_NAME2ID,
                                                      logger=logger, verbose=verbose)
    test_address, test_labels = crawl_directory_dataset(config.dataset.test_path, label_map_dict=EMOTION_NAME2ID,
                                                        logger=logger, verbose=verbose)

    train_dataset = CustomDataset(train_address, train_labels, transform=train_transform, logger=logger)
    val_dataset = CustomDataset(val_address, val_labels, transform=val_transform, logger=logger)
    test_dataset = CustomDataset(test_address, test_labels, transform=test_transform, logger=logger)

    train_loader = DataLoader(train_dataset, config.dataset.batch_size, config.dataset.train_shuffle,
                              num_workers=config.dataset.num_workers, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, config.dataset.batch_size, config.dataset.val_shuffle,
                            num_workers=config.dataset.num_workers, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, config.dataset.batch_size, config.dataset.test_shuffle,
                             num_workers=config.dataset.num_workers, collate_fn=test_dataset.collate_fn)
    if dataloader_name:
        if dataloader_name == 'train':
            loader = train_loader
        elif dataloader_name == 'val':
            loader = val_loader
        elif dataloader_name == 'test':
            loader = test_loader
        else:
            value_error_log(logger, f"{dataloader_name} is not valid")
        return loader
    return train_loader, val_loader, test_loader
