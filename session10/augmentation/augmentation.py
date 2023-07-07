import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train data transformations
train_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=36, min_width=36, always_apply=True, p=0.5),
        A.RandomCrop(height=32, width=32, always_apply=True, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.dropout.cutout.Cutout (num_holes=8, max_h_size=8, max_w_size=8, fill_value=0.4733, always_apply=False, p=0.5),
        A.Normalize(mean=(0.4913, 0.4821, 0.4465), std=(0.2470, 0.2434, 0.2615),p=1.0),
        ToTensorV2(),
    ]
)

# Test data transformations
test_transforms = A.Compose(
    [
        A.Normalize(mean=(0.4913, 0.4821, 0.4465), std=(0.2470, 0.2434, 0.2615),p=1.0),
        ToTensorV2(),

    ]
)
