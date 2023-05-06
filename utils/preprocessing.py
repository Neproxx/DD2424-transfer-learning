import os
import tarfile
from torchvision.utils import save_image
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image, ImageReadMode

def extract_if_not_already(tar_path: str, dlabel: str ="train", variant: str = None):
    """
    Extracts the data from compressed tar files and saves them into either
    the "data/<raw|preprocessed>/<train|test>", depending on the whether the
    variant is "raw" or "preprocessed" and the dataset label is "train" or "test".
    If the variant is None, the data is saved to "data/<train|test>".
    """
    base_path = os.path.join("data", variant) if variant else "data"
    if not os.path.exists(os.path.join(base_path, dlabel)):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(base_path)

def preprocess_images(dlabel: str = "train",
                      resolution: tuple = (224, 224),
                      n_per_class: int = 60
                      ):
    """
    Preprocesses the images and saves them to the preprocessed_path directory.
    Images are cropped to be square and resized to the specified resolution 224x224.

    Parameters:
        dataset_label (str): The dataset to preprocess. Either "train" or "test".
        resolution (tuple): The resolution to resize the images to.
        n_per_class (int): The number of images to preprocess per class.
    """
    raw_path = os.path.join("data", "raw", dlabel)
    preprocessed_path = os.path.join("data", "preprocessed")
    classes = os.listdir(raw_path)

    print(f"Preprocessing {len(classes)} classes from {dlabel} set...")
    for class_label in classes:
        class_path_raw = os.path.join(raw_path, class_label)
        class_path_prep = os.path.join(preprocessed_path, dlabel, class_label)
        if not os.path.exists(class_path_prep):
            os.makedirs(class_path_prep)

        fnames = os.listdir(class_path_raw)[:n_per_class]
        for fname in fnames:
            # Load image as PyTorch tensor with dimensions C x H x W
            img = read_image(
                path=os.path.join(class_path_raw, fname),
                mode=ImageReadMode.RGB
                )

            # Not all images are square, so we might need to crop them
            # so that the final, resized image is not distorted
            c, h, w = img.shape
            img = center_crop(img, output_size=min(w, h))

            # Resize uses BILINEAR interpolation mode by default, which
            # simply linearly interpolates between pixels if needed
            img = resize(img, size=resolution)

            # Rescale to [0, 1]
            img = img.float() / 255.0

            save_image(img, os.path.join(class_path_prep, fname))
