import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop, resize

from datasets import load_dataset
from transformers import ConvNextForImageClassification

# Make parent directory visible to import from core
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import VGGFace2Dataset
from core.preprocessing import extract_if_not_already

# Note that in order to use the VGGFace2Dataset dataset, you need to first download the data
# and pass the path to the dataset constructor. Given that the compressed files lie in the
# "./data" folder, you might run the following code to extract the data to
# "./data/preprocessed/train" and "./data/preprocessed/test".

# The files can be downloaded via this google Drive link:
# https://drive.google.com/drive/folders/1pzyStORojl4jei89uto6lXZEMTuzoKSQ?usp=sharing

### Unpack the tar files
# variant=preprocessed means that the images will be extracted to "./data/preprocessed"
# if your set variant=None, they will be extracted to "./data"
train_prep_tar_path = os.path.join("data", "vggface2_train_prep.tar.gz")
test_prep_tar_path = os.path.join("data", "vggface2_test_prep.tar.gz")
extract_if_not_already(test_prep_tar_path, dlabel="test", variant="preprocessed")
extract_if_not_already(train_prep_tar_path, dlabel="train", variant="preprocessed")

### Select any variant of the dataset you want to use, here we choose the original one
train_path = os.path.join("data", "preprocessed", "train")
original_train = VGGFace2Dataset(root_dir=train_path, version="original")
# rotated_train = VGGFace2Dataset(root_dir=train_path, version="rotated")
# grayscale_train = VGGFace2Dataset(root_dir=train_path, version="grayscale")
# rotated_grayscale_train = VGGFace2Dataset(root_dir=train_path, version="rotated_grayscale")

test_path = os.path.join("data", "preprocessed", "test")
original_test = VGGFace2Dataset(root_dir=test_path, version="original")
# rotated_test = VGGFace2Dataset(root_dir=test_path, version="rotated")
# grayscale_test = VGGFace2Dataset(root_dir=test_path, version="grayscale")
# rotated_grayscale_test = VGGFace2Dataset(root_dir=test_path, version="rotated_grayscale")


### Download and query the model
# In this example, we use the tiny model instead of the base model, just because it is faster
# Check the documentation of __getitem__ in lib/dataset.py to see what each dataset variant returns exactly
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
original_dataloader = DataLoader(original_train, batch_size=32, shuffle=True)

for batch in original_dataloader:
    # torch.no_grad(), because we are not training and thus don't want to track gradients
    with torch.no_grad():
        logits = model(batch["image"]).logits
        y_pred = logits.argmax(-1)
        y_pred_labels = [model.config.id2label[y.item()] for y in y_pred]
        print("Predicted labels for the first batch of data:")
        print(y_pred_labels)
        print("")
        break

### Sanity check
# I could not find a class for "faces" in the imagenet dataset, so we want to sanity check
# if the model can at least predict a cat (which it does successfully):
from torchvision.transforms.functional import pil_to_tensor

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
img_prepped = pil_to_tensor(image)
img_prepped = center_crop(img_prepped, 224)
img_prepped = resize(img_prepped, 224)
img_prepped = img_prepped.float() / 255.0
with torch.no_grad():
    outputs = model(img_prepped.unsqueeze(0))
    predicted_label = outputs.logits[0].argmax(-1).item()

    print("")
    print("Predicted label for the cat image:")
    print(model.config.id2label[predicted_label])

# NOTE: Even though everything seems to work, I am not 100% confident that the data
# is in exactly the correct format, we may want to compare what happens if we use the
# pre-defined image processor from the library and see if performance improves.
