import os
from glob import glob

import torch.utils.data as data
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path):
    images = []
    if os.path.isdir(path):
        file_list = glob(path + '/*')
        for item in file_list:
            images.append((item))

    else:
        images.append((path))

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        imgs = make_dataset(root)

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return (path, img)

    def __len__(self):
        return len(self.imgs)
