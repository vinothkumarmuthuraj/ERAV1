import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(image_size, image, bboxes):
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(image_size)),
            A.PadIfNeeded(min_height=int(image_size), min_width=int(image_size), border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=10, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
            A.RandomCrop(width=image_size, height=image_size),
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
            A.OneOf([A.ShiftScaleRotate(rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT)]),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Posterize(p=0.1),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2()],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ))

    transforms = train_transforms(image=image, bboxes=bboxes)
    return transforms

def get_test_transforms(image_size, image, bboxes):
    test_transforms = A.Compose([A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2()],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]))

    transforms = test_transforms(image=image, bboxes=bboxes)
    return transforms
