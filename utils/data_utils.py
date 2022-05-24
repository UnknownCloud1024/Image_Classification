# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import json
import random

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms


def read_split_data(root: str, val_rate: float = 0.2, plot_image: bool = True):
    # Ensure that the random results can be reproduced
    random.seed(0)
    assert os.path.exists(root), f'dataset root {root} does not exist.'

    # Traverse folders. A folder corresponds to a class.
    flower_classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    # Sort to ensure consistent order
    flower_classes.sort()

    # Encode the classes and generate the corresponding index
    class_indices = dict((k, v) for v, k in enumerate(flower_classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    # Training set: Path and corresponding index information of all pictures
    train_images_path, train_images_label = [], []

    # Validation set: Path and corresponding index information of all pictures
    val_images_path, val_images_label = [], []

    # Total number of samples per class
    every_class_num = []

    # Supported picture formats
    images_format = [".jpg", ".JPG", ".png", ".PNG"]

    # Traverse the files under each folder
    for cla in flower_classes:
        cla_path = os.path.join(root, cla)

        # Get the path of all pictures under each class folder
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in images_format]

        # Get the index corresponding to the class
        image_class = class_indices[cla]

        # Gets the number of samples in this class
        every_class_num.append(len(images))

        # Proportional random sampling validation set
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print(f"{sum(every_class_num)} images found in dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")

    if plot_image:
        plt.bar(range(len(flower_classes)), every_class_num, align='center')
        plt.xticks(range(len(flower_classes)), flower_classes)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


class MyDataSet(Dataset):
    """MyDataset"""

    def __init__(self, images_path: list, images_label: list, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError(f"image: {self.images_path[item]} is not RGB mode")
        label = self.images_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def get_dataset_dataloader(data_path, batch_size):
    train_images_path, train_iamges_label, val_images_path, val_images_label = read_split_data(root=data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),

        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_label=train_iamges_label,
                              transform=data_transform['train'])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_label=val_images_label,
                            transform=data_transform['val'])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f"Using {nw} dataloader workers every process.")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )

    return train_dataset, val_dataset, train_dataloader, val_dataloader