import os
import tarfile
from core.preprocessing import extract_if_not_already, preprocess_images

# Specify the path to the raw tar files you downloaded
# Files can be downloaded from here: https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b
train_tar_path = os.path.join("path", "to", "vggface2_train.tar.gz")
test_tar_path = os.path.join("path", "to", "vggface2_test.tar.gz")

print("Starting to preprocess...")

### Extract raw tar files
extract_if_not_already(test_tar_path, dlabel="test", variant="raw")
extract_if_not_already(train_tar_path, dlabel="train", variant="raw")

### Preprocess images
preprocess_images(dlabel="test", n_per_class=60)
preprocess_images(dlabel="train", n_per_class=60)

### Store preprocessed datasets as tar files
preprocessed_path = os.path.join("data", "preprocessed")
train_prep_tar_path = os.path.join("data", "vggface2_train_prep.tar.gz")
test_prep_tar_path = os.path.join("data", "vggface2_test_prep.tar.gz")
with tarfile.open(train_prep_tar_path, "w:gz") as tar:
    tar.add(os.path.join(preprocessed_path, "train"), arcname="train")
with tarfile.open(test_prep_tar_path, "w:gz") as tar:
    tar.add(os.path.join(preprocessed_path, "test"), arcname="test")
print("Done.")
