import os
from pathlib import Path
import shutil
from PIL import Image
from typing import Callable, Optional, Any
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class OxfordPetsDataset(ImageFolder):
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

    def _preprocess(self):
        src_path = os.path.join(self.root, self.folder)
        cat_path = os.path.join(src_path, "cat")
        dog_path = os.path.join(src_path, "dog")
        
        os.makedirs(cat_path, exist_ok=True)
        os.makedirs(dog_path, exist_ok=True)
        
        for image_path in Path(src_path).glob("*.jpg"):
            image_name = image_path.name
            first_letter = image_name[0]
            if first_letter.isupper():
                shutil.move(image_path, os.path.join(cat_path, image_name))
            else:
                shutil.move(image_path, os.path.join(dog_path, image_name))

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

        

