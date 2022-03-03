import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_normal_aug(img_h, img_w, mean, std, **kwargs):
    train_transform = A.Compose(
        [
            A.Resize(height=img_h, width=img_w),
            A.HorizontalFlip(p=0.5),
            A.ToGray(always_apply=True, p=1),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ])
    val_transform = A.Compose(
        [A.Resize(height=img_h, width=img_w),
         A.HorizontalFlip(p=0.5),
         A.ToGray(always_apply=True, p=1),
         A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
         ToTensorV2()
         ])
    test_transform = A.Compose(
        [A.Resize(height=img_h, width=img_w),
         A.ToGray(always_apply=True, p=1),
         A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
         ToTensorV2()
         ]
    )
    return train_transform, val_transform, test_transform


def get_augmentation(aug_name, img_h, img_w, mean, std, **kwargs):
    augmentations = dict(normal=get_normal_aug)
    return augmentations[aug_name](img_h=img_h, img_w=img_w, mean=mean, std=std, **kwargs)
