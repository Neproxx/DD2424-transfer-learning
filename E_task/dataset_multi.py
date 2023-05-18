import os
from pathlib import Path
import shutil
from PIL import Image
from typing import Callable, Optional, Any
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class OxfordPetsDatasetMulti(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, loader: Callable[[str], Any] = ..., is_valid_file: Optional[Callable[[str], bool]] = None):
    
        self.root = root
        self.url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        self.filename = "images.tar.gz"
        self.tgz_md5 = "3d0f60868b3e1d440ab0f39d9f9ca8a9"
        self.folder = "images"

        if not os.path.exists(os.path.join(self.root, self.folder)):
            download_and_extract_archive(self.url, self.root, filename=self.filename)
        
        self._preprocess()

        super().__init__(
            root = os.path.join(self.root, self.folder),
            transform=transform,
            target_transform=target_transform,
            loader=pil_loader,
            is_valid_file=is_valid_file,
        )
    
    # define a preprocessing function to split the dataset into 37 classes, for each image the class is the name of the image, excludin the final numbers
    def _preprocess(self):
        src_path = os.path.join(self.root, self.folder)

        for image_path in Path(src_path).glob("*.jpg"):
            image_name = image_path.name
            # remove the extension
            label = image_name.split('.')[0]
            # discard all characters that are not letters
            label = ''.join([i for i in label if not i.isdigit()])
            # remove the final character
            label = label[:-1]
            # create folders for labels if they don't exist
            os.makedirs(os.path.join(src_path, label), exist_ok=True)
            # move the image to the correct folder
            shutil.move(image_path, os.path.join(src_path, label, image_name))


def pil_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_augmentation(flip: bool = False, rotation: bool = False, crop: bool = False):
    transforms_list = []
    if flip:
        transforms_list.append(transforms.RandomHorizontalFlip())
    if rotation:
        transforms_list.append(transforms.RandomRotation(20))
    if crop:
        transforms_list.append(transforms.RandomResizedCrop(224))
    
    return transforms.Compose(transforms_list)