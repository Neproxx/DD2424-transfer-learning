import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate, rgb_to_grayscale
from torchvision.io import read_image, ImageReadMode


def rotate_image(img, label_as_onehot=True):
    """Returns a randomly rotated version of the image"""
    degrees = random.choice([0, 90, 180, 270])
    img = rotate(img, degrees)
    if label_as_onehot:
        label = torch.zeros(4)
        label[degrees // 90] = 1
        return img, label
    return img, degrees


class VGGFace2Dataset(Dataset):
    def __init__(self, root_dir: str, version: str = "original"):
        """
        Parameters:
            root_dir (str): The directory containing the preprocessed images, e.g.
                "data/preprocessed/train" or "data/preprocessed/test".
            version (str): The version of the dataset to use.
                Either "original", "rotated", "grayscale" or "rotated_grayscale".
        """
        self.root_dir = root_dir
        self.version = version
        self.classes = os.listdir(root_dir)
        self.n_classes_orig = len(self.classes)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        # Normalization parameters provided by VGGFace2 authors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # Store info about output dimensions of this specific version of the dataset
        self.output_size = self._get_output_size()

        # Create list of tuples (filename, class_id) that contains all images
        self.filenames = []
        for c in self.classes:
            class_path = os.path.join(root_dir, c)
            fnames_of_class = os.listdir(class_path)
            self.filenames += [
                (os.path.join(c, f), self.class_to_idx[c]) for f in fnames_of_class
            ]

    def __len__(self):
        return len(self.filenames)

    def _to_onehot(self, class_id):
        onehot = torch.zeros(self.n_classes_orig)
        onehot[class_id] = 1
        return onehot

    def _get_output_size(self):
        if self.version == "original":
            return self.n_classes_orig
        elif self.version == "rotated":
            return 4
        elif self.version == "grayscale":
            return 3 * 224 * 224
        elif self.version == "rotated_grayscale":
            return 4 + 3 * 224 * 224
        else:
            return None

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset as a dictionary with keys "image" and "label".
        The image is a PyTorch tensor with dimensions C x H x W, where C is the number of
        channels and corresponds to 3 for RGB and to 1 for grayscale.
        The label depends on the version of the dataset:
            - "original": The class label as a onehot vector.
            - "rotated": The rotation of the image as a onehot vector of length 4.
            - "grayscale": The original but flattened image.
            - "rotated_grayscale": The first four elements represent the rotation of the image as
            a onehot vector of length 4, the remaining elements are the flattened original image.
        """
        fname, class_id = self.filenames[idx]
        img_orig = read_image(
            path=os.path.join(self.root_dir, fname), mode=ImageReadMode.RGB
        )

        # re-scale to [0,1] and convert from byte to float tensor
        img_orig = img_orig.float() / 255.0

        # Normalize using values provided by VGGFace2 authors
        # NOTE: I (Marcel) am not sure if the pre-processing steps of the model itself
        # should be applied unchanged or if normalizing should be based on the actual
        # finetune image dataset at hand.
        img_orig = (img_orig - self.mean) / self.std

        sample = {"image": img_orig, "label": self._to_onehot(class_id)}

        # Generate different versions of this sample for different tasks
        if self.version == "rotated" or self.version == "rotated_grayscale":
            rotated_image, rotation_degrees = rotate_image(
                img_orig, label_as_onehot=True
            )
            sample["image"] = rotated_image
            sample["label"] = rotation_degrees
        if self.version == "grayscale" or self.version == "rotated_grayscale":
            # Gray scale image is 1-channel, but we need 3-channel for the model
            image_gray = rgb_to_grayscale(sample["image"])
            sample["image"] = torch.cat((image_gray, image_gray, image_gray), dim=0)
            sample["label"] = img_orig.reshape(-1)
        if self.version == "rotated_grayscale":
            sample["label"] = torch.cat((rotation_degrees, img_orig.reshape(-1)))

        return sample
