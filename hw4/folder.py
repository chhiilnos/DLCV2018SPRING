import torch.utils.data as data
import pandas as pd
from PIL import Image
import os
import os.path

def make_dataset(root):
    images = []
    root = os.path.expanduser(root)
    df = pd.read_csv(os.path.join(root,'train.csv'))
    file_list = df["image_name"].tolist()
    label_list = df["Male"].tolist()
    for idx, fname in enumerate(file_list):
      path = os.path.join(root,'train',fname)
      item = (path, label_list[idx])
      images.append(item)
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class Face(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/train/xxx.png
        root/train/xxy.png
        root/train/xxz.png
        root/test/123.png
        root/test/nsdf3.png
        root/test/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
