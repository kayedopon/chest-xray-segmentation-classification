from torchvision import transforms


def get_train_transforms(augment=True):
    operations = [
        transforms.Resize((224, 224))
    ]

    if augment:
        operations.extend([
            transforms.RandomAffine(
                degrees=7,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            )
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(operations)


def get_eval_transforms():
    return get_train_transforms(augment=False)